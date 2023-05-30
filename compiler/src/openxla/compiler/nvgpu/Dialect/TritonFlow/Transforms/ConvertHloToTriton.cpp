// Copyright 2023 The OpenXLA Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "openxla/compiler/nvgpu/Dialect/TritonFlow/Conversion/ConvertHloToTriton.h"

#include "mhlo/IR/hlo_ops.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Types.h"
#include "mlir/Transforms/DialectConversion.h"
#include "openxla/compiler/nvgpu/Dialect/TritonFlow/IR/TritonFlowDialect.h"
#include "openxla/compiler/nvgpu/Dialect/TritonFlow/IR/TritonFlowOps.h"
#include "openxla/compiler/nvgpu/Dialect/TritonFlow/Transforms/PassDetail.h"
#include "openxla/compiler/nvgpu/Dialect/TritonFlow/Transforms/Passes.h"

#define GEN_PASS_DEF_CONVERTHLOTOTRITON
#include "openxla/compiler/nvgpu/Dialect/TritonFlow/Transforms/Passes.h.inc"

using namespace mlir;

namespace openxla::compiler::nvgpu::tritonflow {

class ConvertHloToTriton
    : public ::impl::ConvertHloToTritonBase<ConvertHloToTriton> {
 public:
  void runOnOperation() override {
    auto *context = &getContext();

    TypeConverter typeConverter;
    typeConverter.addConversion([](Type type) { return type; });

    // Ensure all Triton custom calls get lowered to TritonFlow calls.
    ConversionTarget conversionTarget(*context);
    conversionTarget.addLegalDialect<arith::ArithDialect, TritonFlowDialect>();
    conversionTarget.addDynamicallyLegalOp<mhlo::CustomCallOp>(
        [](mhlo::CustomCallOp op) {
          return op.getCallTargetName() != "__triton$call";
        });

    RewritePatternSet patterns(&getContext());
    populateHloToTritonPatterns(typeConverter, patterns);

    if (failed(applyPartialConversion(getOperation(), conversionTarget,
                                      std::move(patterns)))) {
      getOperation().emitError()
          << "conversion from Hlo to TritonFlow dialect failed";
      return signalPassFailure();
    }
  }
};

std::unique_ptr<OperationPass<ModuleOp>> createConvertHloToTritonPass() {
  return std::make_unique<ConvertHloToTriton>();
}

}  // namespace openxla::compiler::nvgpu::tritonflow
