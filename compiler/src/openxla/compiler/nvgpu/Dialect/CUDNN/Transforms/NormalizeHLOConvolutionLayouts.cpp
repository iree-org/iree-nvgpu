// Copyright 2023 The OpenXLA Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "openxla/compiler/nvgpu/Dialect/CUDNN/IR/CUDNNTypes.h"
#include "openxla/compiler/nvgpu/Dialect/CUDNN/Transforms/Passes.h"
#include "stablehlo/dialect/StablehloOps.h"

using namespace mlir;

namespace openxla::compiler::nvgpu::cudnn {

namespace {

#define GEN_PASS_DECL_NORMALIZEHLOCONVOLUTIONLAYOUTSPASS
#define GEN_PASS_DEF_NORMALIZEHLOCONVOLUTIONLAYOUTSPASS
#include "openxla/compiler/nvgpu/Dialect/CUDNN/Transforms/Passes.h.inc"

struct NormalizeConvolutionLayoutPattern
    : public OpRewritePattern<stablehlo::ConvolutionOp> {
  NormalizeConvolutionLayoutPattern(mlir::MLIRContext* ctx, Layout tensorLayout,
                                    Layout kernelLayout)
      : OpRewritePattern<stablehlo::ConvolutionOp>(ctx),
        tensorLayout(tensorLayout),
        kernelLayout(kernelLayout) {}

  LogicalResult matchAndRewrite(stablehlo::ConvolutionOp op,
                                PatternRewriter& rewriter) const override {
    auto opDims = op.getDimensionNumbers();
    auto inputSpatialDimensions = opDims.getInputSpatialDimensions();

    // For now, only NHWC tensor and KCHW kernel layout are supported. The
    // convolution must be of rank 4.
    if (tensorLayout != Layout::NHWC) {
      return rewriter.notifyMatchFailure(
          op, "Only NHWC tensor layout is implemented");
    }
    if (kernelLayout != Layout::KCHW && kernelLayout != Layout::KHWC) {
      return rewriter.notifyMatchFailure(
          op, "Only KCHW and KHWC kernel layout is implemented");
    }
    if (inputSpatialDimensions.size() != 2) {
      return rewriter.notifyMatchFailure(op,
                                         "Expect exactly 2 spatial dimensions");
    }

    // Determine tensor argument permutation.
    SmallVector<int64_t> lhsPermutation(4, -1);
    assert(tensorLayout == Layout::NHWC);
    lhsPermutation[0] = opDims.getInputBatchDimension();
    lhsPermutation[1] = inputSpatialDimensions[0];
    lhsPermutation[2] = inputSpatialDimensions[1];
    lhsPermutation[3] = opDims.getInputFeatureDimension();

    // Determine kernel argument permutation.
    SmallVector<int64_t> rhsPermutation;
    int64_t newKernelInputFeatureDimension, newKernelOutputFeatureDimension;
    SmallVector<int64_t> newKernelSpatialDimensions;
    auto kernelSpatialDimensions = opDims.getKernelSpatialDimensions();
    if (kernelLayout == Layout::KCHW) {
      rhsPermutation = {opDims.getKernelOutputFeatureDimension(),
                        opDims.getKernelInputFeatureDimension(),
                        kernelSpatialDimensions[0], kernelSpatialDimensions[1]};
      newKernelInputFeatureDimension = 1;
      newKernelOutputFeatureDimension = 0;
      newKernelSpatialDimensions = {2, 3};
    } else if (kernelLayout == Layout::KHWC) {
      rhsPermutation = {opDims.getKernelOutputFeatureDimension(),
                        kernelSpatialDimensions[0], kernelSpatialDimensions[1],
                        opDims.getKernelInputFeatureDimension()};
      newKernelInputFeatureDimension = 3;
      newKernelOutputFeatureDimension = 0;
      newKernelSpatialDimensions = {1, 2};
    }

    // Determine result permutation.
    SmallVector<int64_t> resultPermutation(4, -1);
    assert(tensorLayout == Layout::NHWC);
    auto outputSpatialDimensions = opDims.getOutputSpatialDimensions();
    resultPermutation[opDims.getOutputBatchDimension()] = 0;
    resultPermutation[outputSpatialDimensions[0]] = 1;
    resultPermutation[outputSpatialDimensions[1]] = 2;
    resultPermutation[opDims.getOutputFeatureDimension()] = 3;

    // Do not apply if no transpose is needed. The convolution already conforms
    // to the desired layouts.
    SmallVector<int64_t> identityPermuation = {0, 1, 2, 3};
    if (lhsPermutation == identityPermuation &&
        rhsPermutation == identityPermuation &&
        resultPermutation == identityPermuation) {
      return failure();
    }

    // Transpose tensor operand if needed.
    Location loc = op.getLoc();
    Value lhs = op.getLhs();
    if (lhsPermutation != identityPermuation) {
      lhs = rewriter.createOrFold<stablehlo::TransposeOp>(
          loc, lhs, rewriter.getI64TensorAttr(lhsPermutation));
    }

    // Transpose kernel operand if needed.
    Value rhs = op.getRhs();
    if (rhsPermutation != identityPermuation) {
      rhs = rewriter.createOrFold<stablehlo::TransposeOp>(
          loc, rhs, rewriter.getI64TensorAttr(rhsPermutation));
    }

    // Derive new result type.
    auto resultTy = op.getResult().getType().cast<RankedTensorType>();
    auto shape = resultTy.getShape();
    SmallVector<int64_t> newShape = {shape[opDims.getOutputBatchDimension()],
                                     shape[outputSpatialDimensions[0]],
                                     shape[outputSpatialDimensions[1]],
                                     shape[opDims.getOutputFeatureDimension()]};
    auto newResultTy =
        RankedTensorType::get(newShape, resultTy.getElementType());

    // Recreate convolution op.
    int64_t newIOBatchDimension = 0;
    int64_t newIOFeatureDimension = 3;
    SmallVector<int64_t> newIOSpatialDimensions = {1, 2};
    auto newDimsAttr = mlir::stablehlo::ConvDimensionNumbersAttr::get(
        op.getContext(), newIOBatchDimension, newIOFeatureDimension,
        newIOSpatialDimensions, newKernelInputFeatureDimension,
        newKernelOutputFeatureDimension, newKernelSpatialDimensions,
        newIOBatchDimension, newIOFeatureDimension, newIOSpatialDimensions);
    auto newOp = rewriter.create<stablehlo::ConvolutionOp>(
        loc, newResultTy, lhs, rhs, op.getWindowStridesAttr(),
        op.getPaddingAttr(), op.getLhsDilationAttr(), op.getRhsDilationAttr(),
        op.getWindowReversalAttr(), newDimsAttr, op.getFeatureGroupCountAttr(),
        op.getBatchGroupCountAttr(), op.getPrecisionConfigAttr());

    // Transpose result if needed.
    Value result = newOp.getResult();
    if (resultPermutation != identityPermuation) {
      result = rewriter.createOrFold<stablehlo::TransposeOp>(
          loc, result, rewriter.getI64TensorAttr(resultPermutation));
    }

    rewriter.replaceOp(op, result);
    return success();
  }

  Layout tensorLayout, kernelLayout;
};

LogicalResult eliminateIdentityTranspose(stablehlo::TransposeOp op,
                                         PatternRewriter& rewriter) {
  // Eliminate if transpose is the identity.
  for (const auto& it :
       llvm::enumerate(op.getPermutation().getValues<APInt>())) {
    if (it.index() != it.value()) return failure();
  }
  rewriter.replaceOp(op, op.getOperand());
  return success();
}

LogicalResult composeTranspose(stablehlo::TransposeOp op,
                               PatternRewriter& rewriter) {
  // Only apply to chained transposes.
  auto operandOp = op.getOperand().getDefiningOp<stablehlo::TransposeOp>();
  if (!operandOp) return failure();

  // Compose permutations.
  auto operandPermutation = operandOp.getPermutation().getValues<APInt>();
  auto composedPermutation =
      op.getPermutation()
          .mapValues(op.getPermutation().getElementType(),
                     [&](const APInt& index) -> APInt {
                       return operandPermutation[index.getSExtValue()];
                     })
          .cast<DenseIntElementsAttr>();

  // Create composed transpose op.
  rewriter.replaceOpWithNewOp<stablehlo::TransposeOp>(
      op, op.getResult().getType(), operandOp.getOperand(),
      composedPermutation);
  return success();
}

class NormalizeHLOConvolutionLayoutsPass
    : public impl::NormalizeHLOConvolutionLayoutsPassBase<
          NormalizeHLOConvolutionLayoutsPass> {
 public:
  NormalizeHLOConvolutionLayoutsPass() = default;
  explicit NormalizeHLOConvolutionLayoutsPass(Layout tensorLayout,
                                              Layout kernelLayout)
      : tensorLayout(tensorLayout), kernelLayout(kernelLayout) {}

  void runOnOperation() override {
    MLIRContext* ctx = &getContext();

    // Parse layout options if needed.
    if (!tensorLayoutOption.empty()) {
      if (tensorLayoutOption == "NHWC")
        tensorLayout = Layout::NHWC;
      else if (tensorLayoutOption == "NCHW")
        tensorLayout = Layout::NCHW;
      else
        return signalPassFailure();
    }
    if (!kernelLayoutOption.empty()) {
      if (kernelLayoutOption == "KHWC")
        kernelLayout = Layout::KHWC;
      else if (kernelLayoutOption == "KCHW")
        kernelLayout = Layout::KCHW;
      else
        return signalPassFailure();
    }

    // Tensor and kernel layouts must be defined.
    if (!tensorLayout || !kernelLayout) return signalPassFailure();

    // Populate patterns.
    RewritePatternSet patterns(ctx);
    patterns.add<NormalizeConvolutionLayoutPattern>(ctx, *tensorLayout,
                                                    *kernelLayout);
    patterns.add(composeTranspose);
    patterns.add(eliminateIdentityTranspose);

    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      return signalPassFailure();
    }
  }

 private:
  std::optional<Layout> tensorLayout, kernelLayout;
};

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
createNormalizeHLOConvolutionLayoutsPass(Layout tensorLayout,
                                         Layout kernelLayout) {
  return std::make_unique<NormalizeHLOConvolutionLayoutsPass>(tensorLayout,
                                                              kernelLayout);
}

}  // namespace openxla::compiler::nvgpu::cudnn
