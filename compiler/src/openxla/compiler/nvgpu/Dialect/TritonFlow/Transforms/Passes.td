// Copyright 2023 The OpenXLA Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

include "mlir/Pass/PassBase.td"

#ifndef OPENXLA_COMPILER_NVGPU_DIALECT_TRITONFLOW_TRANSFORMS_PASSES_TD_
#define OPENXLA_COMPILER_NVGPU_DIALECT_TRITONFLOW_TRANSFORMS_PASSES_TD_

//===----------------------------------------------------------------------===//
// Transformations on TritonFlow dialect
//===----------------------------------------------------------------------===//

def OutlineTritonCalls :
    Pass<"openxla-nvgpu-outline-triton-calls", "mlir::ModuleOp"> {
  let summary = "Outlines Triton calls into Triton executables";

  let constructor = [{
    openxla::compiler::nvgpu::tritonflow::createOutlineTritonCallsPass()
  }];

  let dependentDialects = [
    "openxla::compiler::nvgpu::tritonflow::TritonFlowDialect",
  ];
}

def RefineTritonAbi :
    Pass<"openxla-nvgpu-refine-triton-abi", "mlir::ModuleOp"> {
  let summary = "Refines Triton executables ABI to be compatible with IREE";

  let constructor = [{
    openxla::compiler::nvgpu::tritonflow::createRefineTritonAbi()
  }];

  let dependentDialects = [
    "openxla::compiler::nvgpu::tritonflow::TritonFlowDialect",
    "mlir::iree_compiler::IREE::HAL::HALDialect",
  ];
}

//===----------------------------------------------------------------------===//
// Conversions to TritonFlow dialect
//===----------------------------------------------------------------------===//

def ConvertHloToTriton :
    Pass<"openxla-nvgpu-convert-hlo-to-triton", "mlir::ModuleOp"> {
  let summary = "Converts HLO Triton custom calls to TritonFlow calls";

  let constructor = [{
    openxla::compiler::nvgpu::tritonflow::createConvertHloToTritonPass()
  }];

  let dependentDialects = [
    "openxla::compiler::nvgpu::tritonflow::TritonFlowDialect",
    "mlir::arith::ArithDialect",
    "mlir::mhlo::MhloDialect",
    "mlir::triton::TritonDialect",
  ];
}

//===----------------------------------------------------------------------===//
// Conversions from TritonFlow dialect
//===----------------------------------------------------------------------===//

def ConvertTritonToFlowDispatch :
    Pass<"openxla-nvgpu-convert-triton-to-flow-dispatch", "mlir::ModuleOp"> {
  let summary = "Converts Triton dispatches to IREE Flow dispatch";

  let constructor = [{
    openxla::compiler::nvgpu::tritonflow::createConvertTritonToFlowDispatchPass()
  }];

  let dependentDialects = [
    "openxla::compiler::nvgpu::tritonflow::TritonFlowDialect",
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
