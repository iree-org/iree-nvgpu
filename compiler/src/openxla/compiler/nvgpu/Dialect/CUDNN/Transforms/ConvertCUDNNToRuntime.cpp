// Copyright 2023 The OpenXLA Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "openxla/compiler/nvgpu/Dialect/CUDNN/Conversion/ConvertCUDNNToRuntime.h"

#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Types.h"
#include "mlir/Transforms/DialectConversion.h"
#include "openxla/compiler/nvgpu/Dialect/CUDNN/IR/CUDNNDialect.h"
#include "openxla/compiler/nvgpu/Dialect/CUDNN/IR/CUDNNTypes.h"
#include "openxla/compiler/nvgpu/Dialect/CUDNN/Transforms/PassDetail.h"
#include "openxla/compiler/nvgpu/Dialect/CUDNN/Transforms/Passes.h"

#define GEN_PASS_DEF_CONVERTCUDNNTORUNTIME
#include "openxla/compiler/nvgpu/Dialect/CUDNN/Transforms/Passes.h.inc"

namespace IREE = mlir::iree_compiler::IREE;

using namespace mlir;

namespace openxla::compiler::nvgpu::cudnn {

class ConverCudnnToRuntime
    : public ::impl::ConvertCudnnToRuntimeBase<ConverCudnnToRuntime> {
 public:
  void runOnOperation() override {
    auto *context = &getContext();

    TypeConverter typeConverter;
    typeConverter.addConversion([](Type type) { return type; });
    typeConverter.addConversion([](CudnnTensorType tensor) {
      return CudnnTensorType::get(tensor.getContext());
    });

    // Ensure all cuDNN dialect operations go away.
    ConversionTarget conversionTarget(*context);
    conversionTarget.addIllegalDialect<cudnn::CUDNNDialect>();
    conversionTarget.addLegalOp<IREE::Flow::TensorAllocaOp>();
    conversionTarget.addLegalOp<IREE::HAL::TensorExportOp>();
    conversionTarget.addLegalDialect<IREE::Util::UtilDialect>();
    conversionTarget.addLegalDialect<func::FuncDialect>();
    conversionTarget.addLegalDialect<arith::ArithDialect>();

    RewritePatternSet patterns(&getContext());
    populateCudnnToRuntimePatterns(typeConverter, patterns);

    if (failed(applyPartialConversion(getOperation(), conversionTarget,
                                      std::move(patterns)))) {
      getOperation().emitError() << "conversion from cuDNN to runtime failed";
      return signalPassFailure();
    }
  }
};

std::unique_ptr<OperationPass<ModuleOp>> createConvertCudnnToRuntimePass() {
  return std::make_unique<ConverCudnnToRuntime>();
}

}  // namespace openxla::compiler::nvgpu::cudnn
