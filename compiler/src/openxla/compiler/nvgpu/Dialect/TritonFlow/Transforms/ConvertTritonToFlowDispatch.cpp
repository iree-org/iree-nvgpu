// Copyright 2023 The OpenXLA Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "openxla/compiler/nvgpu/Dialect/TritonFlow/Conversion/ConvertTritonToFlowDispatch.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Types.h"
#include "mlir/Transforms/DialectConversion.h"
#include "openxla/compiler/nvgpu/Dialect/TritonFlow/Transforms/PassDetail.h"
#include "openxla/compiler/nvgpu/Dialect/TritonFlow/Transforms/Passes.h"

#define GEN_PASS_DEF_CONVERTTRITONTOFLOWDISPATCH
#include "openxla/compiler/nvgpu/Dialect/TritonFlow/Transforms/Passes.h.inc"

namespace IREE = mlir::iree_compiler::IREE;

using namespace mlir;
using namespace mlir::triton;

namespace openxla::compiler::nvgpu::tritonflow {

class ConvertTritonToFlowDispatch
    : public ::impl::ConvertTritonToFlowDispatchBase<
          ConvertTritonToFlowDispatch> {
 public:
  void runOnOperation() override {
    auto *context = &getContext();

    TypeConverter typeConverter;
    typeConverter.addConversion([](Type type) { return type; });

    // Ensure all TritonFlow operations go away.
    ConversionTarget conversionTarget(*context);
    conversionTarget.addIllegalDialect<TritonFlowDialect>();
    conversionTarget.addLegalDialect<IREE::HAL::HALDialect>();
    conversionTarget.addLegalDialect<IREE::Flow::FlowDialect>();

    RewritePatternSet patterns(&getContext());
    populateTritonToFlowDispatchPatterns(typeConverter, patterns);

    if (failed(applyPartialConversion(getOperation(), conversionTarget,
                                      std::move(patterns)))) {
      getOperation().emitError()
          << "conversion from Triton to cusotm dispatch failed";
      return signalPassFailure();
    }
  }
};

std::unique_ptr<OperationPass<ModuleOp>>
createConvertTritonToFlowDispatchPass() {
  return std::make_unique<ConvertTritonToFlowDispatch>();
}

}  // namespace openxla::compiler::nvgpu::tritonflow
