// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "openxla/compiler/nvgpu/Dialect/FlowTransformExtension/IR/FlowTransformExtension.h"

#include "iree-dialects/Transforms/TransformMatchers.h"
#include "iree/compiler/Dialect/Flow/IR/FlowDialect.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/IR/TransformInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define DEBUG_TYPE "nvgpu-flow-transform-extension"
#define DBGS() llvm::dbgs() << "[" DEBUG_TYPE "]"

namespace mlir {
namespace iree_compiler {
namespace IREE {

void transform_dialect::FilterOutAlreadyInDispatchRegionOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  transform::onlyReadsHandle(getTarget(), effects);
  transform::producesHandle(getTransformed(), effects);
  transform::onlyReadsPayload(effects);
}

DiagnosedSilenceableFailure
transform_dialect::FilterOutAlreadyInDispatchRegionOp::apply(
    transform::TransformRewriter &rewriter,
    transform::TransformResults &results, transform::TransformState &state) {
  RaggedArray<Operation *> inputs;
  for (Value operand : getTarget()) {
    inputs.push_back(state.getPayloadOps(operand));
    for (Operation *payload : inputs.at(inputs.size() - 1)) {
      if (!payload->getParentOfType<Flow::DispatchWorkgroupsOp>()) continue;

      for (OpResult result : getResults()) {
        results.set(result, {});
      }
      return DiagnosedSilenceableFailure::success();
    }
  }

  for (OpResult result : getResults()) {
    results.set(result, inputs.at(result.getResultNumber()));
  }

  return DiagnosedSilenceableFailure::success();
}

LogicalResult transform_dialect::FilterOutAlreadyInDispatchRegionOp::verify() {
  if (getNumOperands() != getNumResults()) {
    return emitOpError() << "expects the same number of operands and results";
  }
  return success();
}

///////////////////////////////////////////////////////////////////////////////
//
// The following code is mostly copied from IREE with a minor adaptation:
// the matcher doesn't include `allTileableOpsCaptured` so it can run on the
// graph level instead of within an individual dispatch region.
// TODO: make this optional in IREE.
//
///////////////////////////////////////////////////////////////////////////////

static constexpr int64_t kCudaWarpSize = 32;

static void makeReductionMatcherCopiedFromIREE(
    transform_ext::MatcherContext &matcherContext,
    transform_ext::StructuredOpMatcher *&reductionCapture,
    transform_ext::StructuredOpMatcher *&fillCapture,
    transform_ext::StructuredOpMatcher *&leadingCapture,
    transform_ext::StructuredOpMatcher *&trailingCapture,
    transform_ext::MatchedReductionCaptures &captures) {
  using namespace transform_ext;

  // The core part of the matcher is anchored on a particular reduction op.
  auto &reduction =
      m_StructuredOp(matcherContext)
          // Op has at least a parallel a reduction dimension and at
          // most 3 parallel dimensions.
          // TODO: relax once we have global collapse/expand_shape.
          //
          .rank(NumGreaterEqualTo(2))
          .rank(NumLowerEqualTo(4))
          .rank(CaptureRank(captures.reductionRank))
          // Op has a single most-minor reduction.
          .dim(-1, utils::IteratorType::reduction)
          // Capture op sizes.
          .dim(AllDims(), CaptureDims(captures.reductionOpSizes))
          // All other dimensions are parallel.
          .dim(AllDimsExcept({-1}), utils::IteratorType::parallel)
          // Single input for now, can be arbitrary projected permutations.
          // TODO: Multiple inputs, can be arbitrary projected permutations.
          // TODO: Watch out for multiple inputs though as a reduction turns
          //       into a contraction when mixed with projected
          //       permutations. A reduction is often bandwidth bound but
          //       contraction is a different beast that is compute bound
          //       and has a very different schedule.
          //
          .input(NumEqualsTo(1))
          .input(AllOperands(), IsProjectedPermutation())
          // Single output supported atm.
          // TODO: Multiple outputs.
          //
          .output(NumEqualsTo(1))
          // A reduction output must be a projected permutation, match it but we
          // could also drop this technically.
          .output(AllOperands(), IsProjectedPermutation())
          // Only single combiner for now due to reduction warp
          // distribution.
          // TODO: relax this once reduction distribution is more powerful.
          //
          .output(0, CaptureElementTypeBitWidth(
                         captures.reductionOutputElementalTypeBitWidth))
          .output(0, SingleCombinerReduction());
  reductionCapture = &reduction;

  // Mandatory FillOp must create the unique output of the reduction.
  // TODO: Relax this, as any map, broadcast, transpose should also work.
  //
  auto &fill = m_StructuredOp<linalg::FillOp>(matcherContext);
  reduction = reduction.output(NumEqualsTo(1)).output(0, fill);
  fillCapture = &fill;

  // Optional leading or trailing op can be any map, transpose, broadcast but
  // not reduce or windowing operation for now.
  // It must create the unique input for the reduction.
  // TODO: match more optional leading ops, one per input of the reduction.
  // TODO: careful about multi-output and turning into a contraction.
  //
  transform_ext::StructuredOpMatcher commonLeadingOrTrailing =
      m_StructuredOp<linalg::GenericOp>(matcherContext)
          // All parallel dimensions.
          .dim(AllDims(), utils::IteratorType::parallel)
          // All inputs are any projected permutation.
          .input(AllOperands(), IsProjectedPermutation())
          .output(AllOperands(), IsPermutation())
          // leading and trailing may have 0, 1 or more input as long as they do
          // not come from unmatched ops. This extra constraint is taken care of
          // separately. This is also a noop but we document it.
          // TODO: Base and derived classes, atm this does not compile.
          // .input(NumGreaterEqualTo(0))
          // Single output supported atm.
          // TODO: extend this.
          //
          .output(NumEqualsTo(1));
  // TODO: match more optional leading ops, one per input of the reduction.
  // TODO: careful about multi-output and turning into a contraction.
  //
  auto &leading =
      m_StructuredOp(matcherContext, commonLeadingOrTrailing)
          .rank(CaptureRank(captures.maybeLeadingRank))
          // Capture op sizes.
          .dim(AllDims(), CaptureDims(captures.leadingOpSizes))
          // Capture output elemental type.
          .output(0, CaptureElementTypeBitWidth(
                         captures.maybeLeadingOutputElementalTypeBitWidth));
  reduction = reduction.input(0, leading, OptionalMatch());
  leadingCapture = &leading;

  // Optional trailing can be any map, transpose, broadcast but not reduce or
  // windowing operation for now.
  // It must be fed by the unique input for the reduction.
  // TODO: match more optional leading ops, one per input of the reduction.
  // TODO: careful about multi-output and turning into a contraction.
  //
  auto &trailing =
      m_StructuredOp(matcherContext, commonLeadingOrTrailing)
          .rank(CaptureRank(captures.maybeTrailingRank))
          // Capture op sizes.
          .dim(AllDims(), CaptureDims(captures.trailingOpSizes))
          // Capture output elemental type.
          .output(0, CaptureElementTypeBitWidth(
                         captures.maybeTrailingOutputElementalTypeBitWidth));
  reduction = reduction.result(0, HasAnyUse(), trailing, OptionalMatch());
  trailingCapture = &trailing;
}

static void makeReductionMatcherCopiedFromIREE(
    transform_ext::MatcherContext &context,
    transform_ext::StructuredOpMatcher *&reductionCapture,
    transform_ext::MatchedReductionCaptures &captures) {
  using namespace transform_ext;
  StructuredOpMatcher *fill;
  StructuredOpMatcher *leading;
  StructuredOpMatcher *trailing;
  makeReductionMatcherCopiedFromIREE(context, reductionCapture, fill, leading,
                                     trailing, captures);
}

/// Match callback for a reduction with optional leading and trailing
/// elementwise operations. Matches *the first* occurrence of such a reduction
/// within an op associated with the given handle.
///
/// Input handles:
///
///   - container op, must be associated with one operation.
///
/// Output handles:
///
///   - leading elementwise op, if any;
///   - the "fill" op preceding the reduction;
///   - reduction op;
///   - trailing elementwise op, if any.
///
/// This callback can be used for both exact (mustExactlyMatchAllTileableOps is
/// set) and partial (mustExactlyMatchAllTileableOps is unset) matches. In the
/// former case, an additional constraint is added to the matcher to ensure that
/// it captured all tilable operations in the parent function. In the latter
/// case, no such constraint is added and the matched part is interpreted as a
/// subgraph in a potentially container (graph).
static DiagnosedSilenceableFailure reductionCallbackCopiedFromIREE(
    transform_ext::MatchCallbackResult &res, Location loc,
    const mlir::transform::TransformState &state, ValueRange handles) {
  if (handles.size() != 1 ||
      !llvm::hasSingleElement(state.getPayloadOps(handles[0]))) {
    return emitSilenceableFailure(loc)
           << "expected one handle to one operation";
  }

  transform_ext::StructuredOpMatcher *pattern, *fill, *leading, *trailing;
  transform_ext::MatchedReductionCaptures ignore;
  transform_ext::MatcherContext matcherContext;
  makeReductionMatcherCopiedFromIREE(matcherContext, pattern, fill, leading,
                                     trailing, ignore);

  // TODO: need a mechanism for this to go around the entire IR,
  // potentially with list matches for each group.
  Operation *root = *state.getPayloadOps(handles[0]).begin();

  WalkResult walkResult = root->walk([&](Operation *op) {
    pattern->resetCapture();
    if (!matchPattern(op, *pattern)) return WalkResult::advance();

    // TODO: notify properly.
    LLVM_DEBUG({
      DBGS() << "leading:\n";
      if (leading->getCaptured()) DBGS() << leading->getCaptured() << "\n";
      DBGS() << "fill: " << fill->getCaptured() << "\n";
      DBGS() << "pattern: " << pattern->getCaptured() << "\n";
      DBGS() << "trailing:\n";
      if (trailing->getCaptured()) DBGS() << trailing->getCaptured() << "\n";
    });

    res.addPotentiallyEmptyPayloadGroup(leading->getCaptured());
    res.addPayloadGroup({fill->getCaptured()});
    res.addPayloadGroup({pattern->getCaptured()});
    res.addPotentiallyEmptyPayloadGroup(trailing->getCaptured());
    return WalkResult::interrupt();
  });

  if (walkResult.wasInterrupted())
    return DiagnosedSilenceableFailure::success();
  return emitSilenceableFailure(loc) << "failed to match";
}

///////////////////////////////////////////////////////////////////////////////
//
// END of code adapted from IREE.
//
///////////////////////////////////////////////////////////////////////////////

void transform_dialect::RegisterNVGPUMatchCallbacks::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  transform::onlyReadsPayload(effects);
}

DiagnosedSilenceableFailure
transform_dialect::RegisterNVGPUMatchCallbacks::apply(
    transform::TransformRewriter &rewriter,
    transform::TransformResults &results, transform::TransformState &state) {
  auto *registry = state.getExtension<transform_ext::MatchCallbacksRegistry>();
  if (!registry) {
    registry = &state.addExtension<transform_ext::MatchCallbacksRegistry>();
  }

  registry->registerCallback("reduction_partial_nvgpu",
                             reductionCallbackCopiedFromIREE);
  return DiagnosedSilenceableFailure::success();
}

}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir

namespace {
/// A Transform dialect extension providing additional transforms operating on
/// the Flow level.
class FlowExtension
    : public mlir::transform::TransformDialectExtension<FlowExtension> {
 public:
  FlowExtension() {
    declareGeneratedDialect<mlir::iree_compiler::IREE::Flow::FlowDialect>();

    registerTransformOps<
#define GET_OP_LIST
#include "openxla/compiler/nvgpu/Dialect/FlowTransformExtension/IR/FlowTransformExtension.cpp.inc"
        >();
  }
};
}  // namespace

void mlir::openxla::nvgpu::registerFlowTransformExtension(
    DialectRegistry &registry) {
  registry.addExtensions<FlowExtension>();
}

#define GET_OP_CLASSES
#include "openxla/compiler/nvgpu/Dialect/FlowTransformExtension/IR/FlowTransformExtension.cpp.inc"
