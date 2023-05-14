// Copyright 2023 The OpenXLA Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

include "mlir/Pass/PassBase.td"

#ifndef OPENXLA_COMPILER_NVGPU_DIALECT_TRITONFLOW_TRANSFORMS_PASSES_TD_
#define OPENXLA_COMPILER_NVGPU_DIALECT_TRITONFLOW_TRANSFORMS_PASSES_TD_

def ConvertTritonToFlowDispatch :
    Pass<"openxla-nvgpu-convert-triton-to-flow-dispatch", "mlir::ModuleOp"> {
  let summary = "Converts Triton dispatches to IREE Flow dispatch";

  let constructor = [{
    openxla::compiler::nvgpu::triton::createConvertTritonToFlowDispatchPass()
  }];

  let dependentDialects = [
    "openxla::compiler::nvgpu::triton::TritonFlowDialect",
    "mlir::iree_compiler::IREE::Flow::FlowDialect",
    "mlir::iree_compiler::IREE::HAL::HALDialect",
    "mlir::LLVM::LLVMDialect",
    "mlir::NVVM::NVVMDialect",
    "mlir::ROCDL::ROCDLDialect",
    "mlir::triton::TritonDialect",
    "mlir::triton::gpu::TritonGPUDialect",
  ];
}

#endif // OPENXLA_COMPILER_NVGPU_DIALECT_TRITONFLOW_TRANSFORMS_PASSES_TD_
