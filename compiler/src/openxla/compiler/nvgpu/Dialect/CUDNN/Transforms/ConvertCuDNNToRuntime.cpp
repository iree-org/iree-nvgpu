// Copyright 2023 The OpenXLA Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <openxla/compiler/nvgpu/Dialect/CUDNN/Conversion/ConvertCuDNNToRuntime.h>
#include <openxla/compiler/nvgpu/Dialect/CUDNN/IR/CUDNNDialect.h>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Transforms/DialectConversion.h"
#include "openxla/compiler/nvgpu/Dialect/CUDNN/Transforms/PassDetail.h"
#include "openxla/compiler/nvgpu/Dialect/CUDNN/Transforms/Passes.h"

#define GEN_PASS_DEF_CONVERTCUDNNTORUNTIME
#include "openxla/compiler/nvgpu/Dialect/CUDNN/Transforms/Passes.h.inc"

using namespace mlir;

namespace openxla::compiler::nvgpu::cudnn {

class ConvertCuDNNToRuntime
    : public ::impl::ConvertCuDNNToRuntimeBase<ConvertCuDNNToRuntime> {
 public:
  void runOnOperation() override {
    auto *context = &getContext();

    TypeConverter typeConverter;
    typeConverter.addConversion([](Type type) { return type; });

    // Ensure all cuDNN dialect operations go away.
    ConversionTarget conversionTarget(*context);
    conversionTarget.addIllegalDialect<cudnn::CUDNNDialect>();
    conversionTarget.addLegalDialect<func::FuncDialect>();

    RewritePatternSet patterns(&getContext());
    populateCuDNNToRuntimePatterns(typeConverter, patterns);

    if (failed(applyFullConversion(getOperation(), conversionTarget,
                                   std::move(patterns)))) {
      getOperation().emitError() << "conversion from cuDNN to runtime failed";
      return signalPassFailure();
    }
  }
};

std::unique_ptr<OperationPass<ModuleOp>> createConvertCuDNNToRuntimePass() {
  return std::make_unique<ConvertCuDNNToRuntime>();
}

}  // namespace openxla::compiler::nvgpu::cudnn
