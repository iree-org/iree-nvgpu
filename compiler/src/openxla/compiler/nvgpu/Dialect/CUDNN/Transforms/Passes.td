// Copyright 2023 The OpenXLA Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

include "mlir/Pass/PassBase.td"

#ifndef OPENXLA_COMPILER_NVGPU_DIALECT_CUDNN_TRANSFORMS_PASSES_TD_
#define OPENXLA_COMPILER_NVGPU_DIALECT_CUDNN_TRANSFORMS_PASSES_TD_

def ExpandCudnnOperations :
    Pass<"openxla-nvgpu-expand-cudnn-operations", "mlir::ModuleOp"> {
  let summary = "Expands cuDNN operations to primitive cuDNN operations";

  let description = [{
    Some cuDNN operations are not supported by the cuDNN runtime fusion engine,
    however they can be expanded into a sequence of primitive operations, that
    could be compiled by cuDNN at runtime.

    For example batch normalization is not supported as a convolution epilogue,
    however at inference phase it can be expanded into a sequence of pointwise
    operations that can be fused into the preceeding convolution.
  }];

  let constructor = [{
    ::openxla::compiler::nvgpu::cudnn::createExpandCudnnOperationsPass()
  }];

  let dependentDialects = [
    "::openxla::compiler::nvgpu::cudnn::CUDNNDialect",
  ];
}

def ConvertCudnnToRuntime :
    Pass<"openxla-nvgpu-convert-cudnn-to-runtime", "mlir::ModuleOp"> {
  let summary = "Converts cuDNN operations to cuDNN runtime calls";

  let constructor = [{
    ::openxla::compiler::nvgpu::cudnn::createConvertCudnnToRuntimePass()
  }];

  let dependentDialects = [
    "::openxla::compiler::nvgpu::cudnn::CUDNNDialect",
    "::mlir::iree_compiler::IREE::HAL::HALDialect",
    "::mlir::func::FuncDialect",
  ];
}

#endif // OPENXLA_COMPILER_NVGPU_DIALECT_CUDNN_TRANSFORMS_PASSES_TD_
