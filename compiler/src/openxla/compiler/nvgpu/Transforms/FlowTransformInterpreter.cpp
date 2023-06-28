// Copyright 2023 The OpenXLA Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-dialects/Dialect/LinalgTransform/StructuredTransformOpsExt.h"
#include "iree/compiler/Dialect/Flow/TransformExtensions/FlowExtensions.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/TransformOps/LinalgMatchOps.h"
#include "mlir/Dialect/Linalg/TransformOps/LinalgTransformOps.h"
#include "mlir/Dialect/Transform/IR/TransformOps.h"
#include "mlir/Dialect/Transform/Transforms/TransformInterpreterPassBase.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Verifier.h"
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

/// Build a named transform sequence with the given names and argument/result
/// types. `inputsConsumed` specifies whether to mark the input as consumed
/// (true) or readonly (false) and must have as many elements as `inputs`.
// TODO: add a builder upstream.
static transform::NamedSequenceOp buildNamedSequence(
    ImplicitLocOpBuilder &builder, StringRef name, ArrayRef<Type> inputs,
    ArrayRef<bool> inputsConsumed, ArrayRef<Type> results) {
  assert(inputs.size() == inputsConsumed.size() &&
         "expected the same number of consumption flags as input types");

  std::string visibilityString;
  llvm::raw_string_ostream os(visibilityString);
  os << SymbolOpInterface::Visibility::Private;
  os.flush();

  auto unitAttr = builder.getUnitAttr();
  auto consumedAttr =
      builder.getStringAttr(transform::TransformDialect::kArgConsumedAttrName);
  auto readonlyAttr =
      builder.getStringAttr(transform::TransformDialect::kArgReadOnlyAttrName);
  auto argAttrs = llvm::to_vector(
      llvm::map_range(inputsConsumed, [&](bool isConsumed) -> Attribute {
        return builder.getDictionaryAttr(
            NamedAttribute(isConsumed ? consumedAttr : readonlyAttr, unitAttr));
      }));

  auto sequence = builder.create<transform::NamedSequenceOp>(
      name, TypeAttr::get(builder.getFunctionType(inputs, results)),
      builder.getStringAttr(visibilityString), builder.getArrayAttr(argAttrs),
      ArrayAttr());
  return sequence;
}

/// Builds transform IR matching the reduction for future dispatch region
/// formation.
static transform::NamedSequenceOp buildReductionDispatchMatcher(
    ImplicitLocOpBuilder &builder, bool emitRemarkOnMatch = false) {
  auto anyOp = builder.getType<transform::AnyOpType>();
  SmallVector<Type> matchedTypes(4, anyOp);
  transform::NamedSequenceOp sequence = buildNamedSequence(
      builder, "match_reduction_partial_nvgpu", {anyOp}, {false}, matchedTypes);

  OpBuilder::InsertionGuard insertionGuard(builder);
  Block *body = sequence.addEntryBlock();
  builder.setInsertionPointToEnd(body);
  auto matched = builder.create<transform_ext::MatchCallbackOp>(
      matchedTypes, "reduction_partial_nvgpu",
      transform::FailurePropagationMode::Propagate, body->getArgument(0));

  auto filtered =
      builder.create<transform_dialect::FilterOutAlreadyInDispatchRegionOp>(
          matchedTypes, matched->getResults());

  Value reductionH = filtered->getResults().drop_back().back();

  if (emitRemarkOnMatch) {
    builder.create<transform_ext::EmitRemarkOp>(reductionH,
                                                "dispatch matched reduction");
  }

  builder.create<transform::YieldOp>(filtered.getResults());
  return sequence;
}

/// Builds transform IR creating a dispatch region for a reduction.
static transform::NamedSequenceOp buildReductionDispatchFormation(
    ImplicitLocOpBuilder &builder) {
  auto anyOp = builder.getType<transform::AnyOpType>();
  SmallVector<Type> matchedTypes(4, anyOp);
  transform::NamedSequenceOp sequence =
      buildNamedSequence(builder, "dispatch_reduction_partial_nvgpu",
                         matchedTypes, {true, true, true, true}, {});

  // This may work because we are always moving _preceding_ ops into the
  // region. Moving an immediately succeeding op would break the walk by
  // invalidating the iterator. Also, the actual implementation in the Flow
  // dialect _clones and erases_, rather than actually _moves_, which can lead
  // to accidental pointer reuse and other memory problems.
  OpBuilder::InsertionGuard insertionGuard(builder);
  Block *body = sequence.addEntryBlock();
  builder.setInsertionPointToEnd(body);
  Value reductionH = body->getArguments().drop_back().back();
  Value trailingH = body->getArguments().back();
  auto [firstH, restH] =
      buildSelectFirstNonEmpty(builder, trailingH, reductionH);
  Value regionH =
      builder.create<transform_dialect::WrapInDispatchRegionOp>(anyOp, firstH);
  SmallVector<Value> handlesToMerge(body->getArguments().begin(),
                                    std::prev(body->getArguments().end(), 2));
  handlesToMerge.push_back(restH);
  Value mergedHandlesH = builder.create<transform::MergeHandlesOp>(
      handlesToMerge, /*deduplicate=*/false);
  regionH =
      builder.create<transform_dialect::MovePrecedingOpIntoDispatchRegionOp>(
          regionH.getType(), mergedHandlesH, regionH);
  builder.create<transform_dialect::RegionToWorkgroupsOp>(anyOp, regionH);

  builder.create<transform::YieldOp>();
  return sequence;
}

/// Builds a transform dialect matcher sequence with the given name that matches
/// a `linalg.matmul` with the given shape.
// TODO: restrict the elemental type to be f32.
static transform::NamedSequenceOp buildMatmulMatchers(
    ImplicitLocOpBuilder &builder, StringRef name, ArrayRef<int64_t> sizes) {
  auto anyOp = builder.getType<transform::AnyOpType>();
  auto matchFunction =
      buildNamedSequence(builder, name, {anyOp}, {false}, {anyOp});

  OpBuilder::InsertionGuard guard(builder);
  Block *funcBlock = matchFunction.addEntryBlock();
  builder.setInsertionPointToEnd(funcBlock);

  auto matchStructured = builder.create<transform::MatchStructuredOp>(
      TypeRange(anyOp), funcBlock->getArgument(0),
      builder.getAttr<transform::FailurePropagationModeAttr>(
          transform::FailurePropagationMode::Propagate));

  {
    OpBuilder::InsertionGuard nestedGuard(builder);

    Block *matcherRegion = builder.createBlock(
        &matchStructured.getBodyRegion(),
        matchStructured.getBodyRegion().begin(), {anyOp}, {builder.getLoc()});
    builder.setInsertionPointToEnd(matcherRegion);
    builder.create<transform::MatchOperationNameOp>(
        matcherRegion->getArgument(0),
        builder.getStrArrayAttr({linalg::MatmulOp::getOperationName()}));

    auto i64Param = builder.getType<transform::ParamType>(builder.getI64Type());
    for (int64_t i = 0, e = sizes.size(); i < e; ++i) {
      Value param =
          builder
              .create<transform::MatchStructuredDimOp>(
                  i64Param, matcherRegion->getArgument(0), ArrayRef<int64_t>(i))
              ->getResult(0);
      Value constant = builder.create<transform::ParamConstantOp>(
          i64Param, builder.getI64IntegerAttr(sizes[i]));
      builder.create<transform::MatchParamCmpIOp>(
          param, constant, transform::MatchCmpIPredicate::eq);
    }
    builder.create<transform::MatchStructuredYieldOp>(
        matcherRegion->getArgument(0));
  }

  builder.create<transform::YieldOp>(matchStructured.getResults());
  return matchFunction;
}

/// Builds a transform dialect sequence with the given name that pads a matmul
/// and performs split-k (reduction splitting) if `splitK` is set.
static transform::NamedSequenceOp buildMatmulPadOptionalSplitK(
    ImplicitLocOpBuilder &builder, StringRef name, bool splitK) {
  auto anyOp = builder.getType<transform::AnyOpType>();
  auto padFunction = buildNamedSequence(builder, name, {anyOp}, {true}, {});

  OpBuilder::InsertionGuard guard(builder);
  Block *funcBlock = padFunction.addEntryBlock();
  builder.setInsertionPointToEnd(funcBlock);

  // TODO: add a nicer builder upstream.
  Attribute f32zero = builder.getF32FloatAttr(0.0);
  auto i64ArrayAttr = [&](ArrayRef<int64_t> values) {
    return builder.getArrayAttr(llvm::to_vector(
        llvm::map_range(values, [&](int64_t value) -> Attribute {
          return builder.getI64IntegerAttr(value);
        })));
  };
  Value paddedH = builder.create<transform::PadOp>(
      anyOp, funcBlock->getArgument(0),
      builder.getArrayAttr({f32zero, f32zero, f32zero}),
      /*padding_values=*/i64ArrayAttr({0, 1, 2}),
      /*pad_to_multiple_of*/ i64ArrayAttr({32, 32, 1728}),
      /*pack_paddings=*/i64ArrayAttr({0, 0, 0}),
      /*transpose_paddings=*/ArrayAttr());

  if (splitK) {
    auto split = builder.create<transform::SplitReductionOp>(
        paddedH, /*splitFactor=*/108, /*insertSplitDimension=*/2);
    builder.create<transform::InterchangeOp>(
        anyOp, split.getSplitLinalgOp(),
        /*iterator_interchange=*/ArrayRef<int64_t>({2, 0, 1, 3}));
  }

  // TODO: localized cleanups.

  builder.create<transform::YieldOp>();
  return padFunction;
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
    additionalSharedTransformModule->get()->setAttr(
        transform::TransformDialect::kWithNamedSequenceAttrName,
        builder.getUnitAttr());
    builder.setInsertionPointToEnd(
        additionalSharedTransformModule.get()->get().getBody());
    constructTransformModule(builder, UnknownLoc::get(context));
    if (failed(mlir::verify(additionalSharedTransformModule->get()))) {
      return failure();
    }

    if (debugPrintConstructedModule) {
      additionalSharedTransformModule->get()->print(llvm::outs());
    }

    return r;
  }

  void runOnOperation() override {
    if (failed(transform::detail::interpreterBaseRunOnOperationImpl(
            getOperation(), getArgument(), additionalSharedTransformModule,
            nullptr,
            /*extraMappings=*/{}, options, transformFileName,
            transformLibraryFileName, debugPayloadRootTag,
            debugTransformRootTag, "iree-opt"))) {
      return signalPassFailure();
    }
  }

  /// Constructs the transform module that contains transformation scripts to be
  /// applied.
  std::optional<LogicalResult> constructTransformModule(OpBuilder &builder,
                                                        Location loc) {
    // Matchers+dispatch builders for each case, ordered by priority.
    SmallVector<Attribute> matchers;
    SmallVector<Attribute> actions;

    auto getSymbolRef = [](transform::NamedSequenceOp sequence) {
      return SymbolRefAttr::get(sequence.getSymNameAttr());
    };

    auto addRewriteRule = [&](transform::NamedSequenceOp match,
                              transform::NamedSequenceOp rewrite) {
      matchers.push_back(getSymbolRef(match));
      actions.push_back(getSymbolRef(rewrite));
    };
    ImplicitLocOpBuilder ib(loc, builder);

    if (enableReductionDispatchFormation) {
      addRewriteRule(buildReductionDispatchMatcher(ib, debugEmitRemarkOnMatch),
                     buildReductionDispatchFormation(ib));
    }

    if (enableMatmulPadSplitK) {
      // Sizes for which padding is enabled.
      SmallVector<std::array<int64_t, 3>> padSizes = {{133, 133, 128}};
      // Sizes for which split-k is enabled.
      SmallVector<std::array<int64_t, 3>> splitKSizes = {{514, 130, 500},
                                                         {515, 131, 512}};

      auto makeMatmulMatcher = [&](ArrayRef<int64_t> sizes) -> Attribute {
        std::string name;
        llvm::raw_string_ostream os(name);
        os << "match_matmul_f32_";
        llvm::interleave(sizes, os, "x");
        os.flush();

        transform::NamedSequenceOp matcher =
            buildMatmulMatchers(ib, name, sizes);
        return SymbolRefAttr::get(matcher.getSymNameAttr());
      };

      llvm::append_range(matchers,
                         llvm::map_range(padSizes, makeMatmulMatcher));
      llvm::append_range(matchers,
                         llvm::map_range(splitKSizes, makeMatmulMatcher));

      transform::NamedSequenceOp padSequence =
          buildMatmulPadOptionalSplitK(ib, "matmul_f32_pad", /*splitK=*/false);
      Attribute padSequenceName =
          SymbolRefAttr::get(padSequence.getSymNameAttr());
      actions.append(padSizes.size(), padSequenceName);

      transform::NamedSequenceOp splitKSequence = buildMatmulPadOptionalSplitK(
          ib, "matmul_f32_split_k", /*splitK=*/true);
      Attribute splitKSequenceName =
          SymbolRefAttr::get(splitKSequence.getSymNameAttr());
      actions.append(splitKSizes.size(), splitKSequenceName);
    }

    if (matchers.empty()) return std::nullopt;

    builder.create<transform::SequenceOp>(
        loc, TypeRange(), transform::FailurePropagationMode::Propagate,
        builder.getType<transform::AnyOpType>(),
        [&](OpBuilder &b, Location loc, Value rootH) {
          ImplicitLocOpBuilder ib(loc, b);
          ib.create<transform_ext::RegisterMatchCallbacksOp>();
          ib.create<transform_dialect::RegisterNVGPUMatchCallbacks>();

          b.create<transform::ForeachMatchOp>(loc, rootH.getType(), rootH,
                                              builder.getArrayAttr(matchers),
                                              builder.getArrayAttr(actions));

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
