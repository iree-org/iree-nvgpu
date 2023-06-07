// Copyright 2023 The OpenXLA Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-dialects/Dialect/LinalgTransform/StructuredTransformOpsExt.h"
#include "iree/compiler/Dialect/Flow/TransformExtensions/FlowExtensions.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Transform/IR/TransformOps.h"
#include "mlir/Dialect/Transform/Transforms/TransformInterpreterPassBase.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Transforms/Passes.h"
#include "openxla/compiler/nvgpu/Dialect/FlowTransformExtension/IR/FlowTransformExtension.h"
#include "openxla/compiler/nvgpu/Transforms/Passes.h"

using namespace mlir;
using namespace mlir::iree_compiler;
using namespace mlir::iree_compiler::IREE;

/// Dynamically selects the first non-empty handle; i.e. if (h1, h2) is:
///   - (non-empty, non-empty), returns (h1, h2)
///   - (empty, non-empty), returns (h2, empty)
///   - (non-empty, empty), returns (h1, empty)
///   - (empty, empty), returns (empty, empty)
/// This is used as a normalization operation that replaces conditionals, either
/// in C++ or in transform IR.
/// This can be thought of as a control-flow -> data-dependent conversion.
static std::pair<Value, Value> buildSelectFirstNonEmpty(ImplicitLocOpBuilder &b,
                                                        Value handle1,
                                                        Value handle2) {
  auto anyOpType = transform::AnyOpType::get(b.getContext());
  auto selector = b.create<transform_ext::TakeFirstOp>(
      anyOpType, anyOpType, ArrayRef<Value>{handle1, handle2});
  return std::make_pair(selector.getFirst(), selector.getRest());
}

/// Builds transform IR forming dispatch regions for reductions.
static void buildReductionDispatch(ImplicitLocOpBuilder &builder, Value scopeH,
                                   bool emitRemarkOnMatch = false) {
  auto anyOp = transform::AnyOpType::get(builder.getContext());
  SmallVector<Type> matchedTypes(4, anyOp);
  auto matched = builder.create<transform_ext::MatchCallbackOp>(
      matchedTypes, "reduction_partial",
      transform::FailurePropagationMode::Suppress, scopeH);

  auto filtered =
      builder.create<transform_dialect::FilterOutAlreadyInDispatchRegionOp>(
          matchedTypes, matched->getResults());

  Value reductionH = filtered->getResults().drop_back().back();
  Value trailingH = filtered->getResults().back();

  if (emitRemarkOnMatch) {
    builder.create<transform_ext::EmitRemarkOp>(reductionH,
                                                "dispatch matched reduction");
  }

  auto [firstH, restH] =
      buildSelectFirstNonEmpty(builder, trailingH, reductionH);
  Value regionH =
      builder.create<transform_dialect::WrapInDispatchRegionOp>(anyOp, firstH);
  SmallVector<Value> handlesToMerge(filtered->getResults().begin(),
                                    std::prev(filtered->getResults().end(), 2));
  handlesToMerge.push_back(restH);
  Value mergedHandlesH = builder.create<transform::MergeHandlesOp>(
      handlesToMerge, /*deduplicate=*/false);
  regionH =
      builder.create<transform_dialect::MovePrecedingOpIntoDispatchRegionOp>(
          regionH.getType(), mergedHandlesH, regionH);
  builder.create<transform_dialect::RegionToWorkgroupsOp>(anyOp, regionH);
}

namespace {

#define GEN_PASS_DECL_FLOWTRANSFORMINTERPRETERPASS
#include "openxla/compiler/nvgpu/Transforms/Passes.h.inc"

#define GEN_PASS_DEF_FLOWTRANSFORMINTERPRETERPASS
#include "openxla/compiler/nvgpu/Transforms/Passes.h.inc"

/// Applies Transform dialect interpreter during the preprocessing stage, at the
/// Flow level. Constructs transforms on-the-fly.
class TransformDialectPreprocessingPass
    : public mlir::transform::TransformInterpreterPassBase<
          TransformDialectPreprocessingPass,
          impl::FlowTransformInterpreterPassBase> {
 public:
  explicit TransformDialectPreprocessingPass(
      StringRef transformFileName = StringRef(),
      StringRef debugPayloadRootTag = StringRef(),
      StringRef debugTransformRootTag = StringRef()) {
    this->transformFileName = transformFileName.str();
    this->debugPayloadRootTag = debugPayloadRootTag.str();
    this->debugTransformRootTag = debugTransformRootTag.str();
  }
  TransformDialectPreprocessingPass(const TransformDialectPreprocessingPass &) =
      default;

  /// Populates the registry with dialects and extensions that may be produced
  /// by this pass.
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<transform::TransformDialect>();
    mlir::openxla::nvgpu::registerFlowTransformExtension(registry);
  }

  /// Additional initialization, constructs the transform module that contains
  /// transformation scripts to be applied.
  mlir::LogicalResult initialize(mlir::MLIRContext *context) override {
    std::shared_ptr<OwningOpRef<ModuleOp>> additionalUnusedTransformModule;
    /*return */ auto r = mlir::transform::detail::interpreterBaseInitializeImpl(
        context, transformFileName, "", additionalSharedTransformModule,
        additionalUnusedTransformModule);

    if (!transformFileName.empty()) return r;

    additionalSharedTransformModule =
        std::make_shared<OwningOpRef<ModuleOp>>(OwningOpRef<ModuleOp>(
            ModuleOp::create(UnknownLoc::get(context), "__transform")));
    OpBuilder builder(context);
    builder.setInsertionPointToEnd(
        additionalSharedTransformModule.get()->get().getBody());
    constructTransformModule(builder, UnknownLoc::get(context));

    return r;
  }

  void runOnOperation() override {
    // This may work because we are always moving _preceding_ ops into the
    // region. Moving an immediately succeeding op would break the walk by
    // invalidating the iterator. Also, the actual implementation in the Flow
    // dialect _clones and erases_, rather than actually _moves_, which can lead
    // to accidental pointer reuse and other memory problems.
    WalkResult walkResult = getOperation()->walk<WalkOrder::PostOrder>(
        [&](linalg::LinalgOp linalgOp) {
          if (failed(transform::detail::interpreterBaseRunOnOperationImpl(
                  linalgOp, getArgument(), additionalSharedTransformModule,
                  nullptr,
                  /*extraMappings=*/{}, options, transformFileName,
                  transformLibraryFileName, debugPayloadRootTag,
                  debugTransformRootTag, "iree-opt"))) {
            return WalkResult::interrupt();
          }
          return WalkResult::advance();
        });
    if (walkResult.wasInterrupted()) return signalPassFailure();
  }

  /// Constructs the transform module that contains transformation scripts to be
  /// applied.
  std::optional<LogicalResult> constructTransformModule(OpBuilder &builder,
                                                        Location loc) {
    builder.create<transform::SequenceOp>(
        loc, TypeRange(), transform::FailurePropagationMode::Propagate,
        builder.getType<transform::AnyOpType>(),
        [&](OpBuilder &b, Location loc, Value rootH) {
          ImplicitLocOpBuilder ib(loc, b);
          ib.create<transform_ext::RegisterMatchCallbacksOp>();
          ib.create<transform_dialect::RegisterNVGPUMatchCallbacks>();

          // Matchers+dispatch builders for each case, ordered by priority.
          buildReductionDispatch(ib, rootH, debugEmitRemarkOnMatch);

          b.create<transform::YieldOp>(loc);
        });
    return success();
  }

 private:
  /// Shared transform module local to the pass. This is a copy of the mechanism
  /// available upstream, but not yet available in IREE.
  // TODO: remove this after IREE updates past LLVM 9d30c6a721edf75d0726e07fb82
  // and openxla-nvgpu updates IREE past the integration point.
  std::shared_ptr<OwningOpRef<ModuleOp>> additionalSharedTransformModule =
      nullptr;
};

}  // namespace

namespace openxla {

std::unique_ptr<mlir::OperationPass<void>>
compiler::nvgpu::createFlowTransformInterpreterPass(
    StringRef transformFileName) {
  return std::make_unique<TransformDialectPreprocessingPass>(transformFileName);
}

}  // namespace openxla
