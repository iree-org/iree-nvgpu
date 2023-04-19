// Copyright 2023 The OpenXLA Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "openxla/compiler/nvgpu/Dialect/CUDNN/Conversion/ConvertCuDNNToRuntime.h"

#include "openxla/compiler/nvgpu/Dialect/CUDNN/IR/CUDNNOps.h"

namespace openxla::compiler::nvgpu {

using namespace mlir;

namespace {

// Converts `cudnn.graph` operations to a function building cuDNN operation
// graph using cuDNN runtime APIs.
struct ConvertCuDNNGraphOp : public OpConversionPattern<cudnn::GraphOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      cudnn::GraphOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    // TODO(ezhulenev): Uh oh! Not implemented!
    return failure();
  }
};

}  // namespace

void populateCuDNNToRuntimePatterns(mlir::MLIRContext *context,
                                    mlir::TypeConverter &typeConverter,
                                    mlir::RewritePatternSet &patterns) {
  patterns.insert<ConvertCuDNNGraphOp>(typeConverter, context);
}

}  // namespace openxla::compiler::nvgpu
