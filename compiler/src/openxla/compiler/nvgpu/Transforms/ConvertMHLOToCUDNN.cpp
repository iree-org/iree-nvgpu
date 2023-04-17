// Copyright 2023 The OpenXLA Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <algorithm>
#include <numeric>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "openxla/compiler/nvgpu/Dialect/CUDNN/IR/CUDNNDialect.h"
#include "openxla/compiler/nvgpu/Dialect/CUDNN/IR/CUDNNOps.h"
#include "openxla/compiler/nvgpu/Transforms/Passes.h"
#include "stablehlo/dialect/StablehloOps.h"

#define GEN_PASS_DEF_CONVERTMHLOTOCUDNN
#include "openxla/compiler/nvgpu/Transforms/Passes.h.inc"

using namespace mlir;

namespace openxla::compiler::nvgpu {

static SmallVector<int64_t> getRowMajorStrides(ArrayRef<int64_t> shape) {
  SmallVector<int64_t> strides(shape.size(), 1);
  std::partial_sum(shape.rbegin(),
                   shape.rend() - 1,
                   strides.rbegin() + 1,
                   std::multiplies<int64_t>());
  return strides;
}

static cudnn::TensorDescType getTensorDescType(TensorType tensor_type) {
  auto shape = tensor_type.getShape();
  Type element_type = tensor_type.getElementType();
  int alignment = 0;
  auto strides = getRowMajorStrides(shape);
  return cudnn::TensorDescType::get(tensor_type.getContext(),
                                    shape, element_type, alignment, strides);
}

namespace {

struct ConvertClamp : public OpRewritePattern<stablehlo::ClampOp> {
  using OpRewritePattern<stablehlo::ClampOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(stablehlo::ClampOp op,
                                PatternRewriter& rewriter) const override {
    llvm::APFloat min = llvm::APFloat::IEEEdouble();
    if (!matchPattern(op.getMin().getDefiningOp(), m_ConstantFloat(&min))) {
      return rewriter.notifyMatchFailure(op, "expected constant min");
    }
    llvm::APFloat max = llvm::APFloat::IEEEdouble();
    if (!matchPattern(op.getMax().getDefiningOp(), m_ConstantFloat(&max)) ||
        !max.isNaN()) {
      return rewriter.notifyMatchFailure(op, "expected NaN max");
    }
    TensorType tensor_type = op.getOperand().getType();
    cudnn::TensorDescType tensor_desc_type = getTensorDescType(tensor_type);
    Value input = rewriter.create<UnrealizedConversionCastOp>(
        op.getLoc(), tensor_desc_type, op.getOperand()).getResult(0);
    Value result = rewriter.create<cudnn::PointWiseReluOp>(
        op.getLoc(), tensor_desc_type, input, tensor_type.getElementType(),
        APFloat(min.convertToDouble()));
    rewriter.replaceOpWithNewOp<UnrealizedConversionCastOp>(
        op, tensor_type, result);
    return success();
  }
};

class ConvertMHLOToCUDNN : public ::impl::ConvertMHLOToCUDNNBase<
    ConvertMHLOToCUDNN> {
 public:
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.insert<ConvertClamp>(&getContext());
    if (failed(::applyPatternsAndFoldGreedily(getOperation(),
                                              std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>> createConvertMHLOToCUDNNPass() {
  return std::make_unique<ConvertMHLOToCUDNN>();
}

} // namespace openxla::compiler::nvgpu
