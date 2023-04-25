// Copyright 2023 The OpenXLA Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <openxla/compiler/nvgpu/Dialect/CUDNN/IR/CUDNNDialect.h>
#include <openxla/compiler/nvgpu/Dialect/CUDNN/IR/CUDNNOps.h>
#include <openxla/compiler/nvgpu/Dialect/CUDNN/IR/CUDNNTypes.h>

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Types.h"
#include "mlir/Transforms/DialectConversion.h"
#include "openxla/compiler/nvgpu/Dialect/CUDNN/Transforms/PassDetail.h"
#include "openxla/compiler/nvgpu/Dialect/CUDNN/Transforms/Passes.h"

#define GEN_PASS_DEF_EXPANDCUDNNOPERATIONS
#include "openxla/compiler/nvgpu/Dialect/CUDNN/Transforms/Passes.h.inc"

namespace IREE = mlir::iree_compiler::IREE;

using namespace mlir;

namespace openxla::compiler::nvgpu::cudnn {

namespace {

struct ExpandBatchNormInferenceOp
    : public OpRewritePattern<BatchNormInferenceOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(BatchNormInferenceOp op,
                                PatternRewriter &rewriter) const final {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    Value x = op.getX();
    Value scale = op.getScale();
    Value offset = op.getOffset();
    Value mean = op.getMean();
    Value var = op.getVariance();
    Value eps = op.getEpsilon();

    auto tensor = x.getType();
    auto vector = var.getType();

    // Use batch norm inference expansion defined by StableHLO:
    // https://github.com/openxla/stablehlo/blob/main/docs/spec.md#batch_norm_inference

    Value centered = b.create<SubOp>(tensor, ValueRange({x, mean}));
    Value stddev = b.create<SqrtOp>(
        tensor, Value(b.create<AddOp>(vector, ValueRange({var, eps}))));
    Value normalized = b.create<DivOp>(tensor, ValueRange({centered, stddev}));
    Value scaled = b.create<MulOp>(tensor, ValueRange({normalized, scale}));
    Value shifted = b.create<AddOp>(tensor, ValueRange({scaled, offset}));

    rewriter.replaceOp(op, shifted);
    return success();
  }
};

}  // namespace

class ExpandCudnnOperations
    : public ::impl::ExpandCudnnOperationsBase<ExpandCudnnOperations> {
 public:
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    ConversionTarget target(getContext());

    patterns.insert<ExpandBatchNormInferenceOp>(patterns.getContext());

    target.addLegalDialect<CUDNNDialect>();
    // TODO(ezhulenev): Operation legality should be defined by the cuDNN graph
    // itself, batch norm itself can be a "root" of the cuDNN graph with
    // pointwise fusion into it.
    target.addIllegalOp<BatchNormInferenceOp>();

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};

std::unique_ptr<OperationPass<ModuleOp>> createExpandCudnnOperationsPass() {
  return std::make_unique<ExpandCudnnOperations>();
}

}  // namespace openxla::compiler::nvgpu::cudnn
