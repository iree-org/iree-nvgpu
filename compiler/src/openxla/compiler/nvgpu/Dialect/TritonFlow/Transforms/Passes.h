// Copyright 2023 The OpenXLA Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef OPENXLA_COMPILER_NVGPU_DIALECT_TRITONFLOW_TRANSFORMS_PASSES_H_
#define OPENXLA_COMPILER_NVGPU_DIALECT_TRITONFLOW_TRANSFORMS_PASSES_H_

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

namespace openxla::compiler::nvgpu::triton {

//===----------------------------------------------------------------------===//
// Conversion from Triton dialect
//===----------------------------------------------------------------------===//

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createConvertTritonToFlowDispatchPass();

//===----------------------------------------------------------------------===//
// OpenXLA Triton passes registration
//===----------------------------------------------------------------------===//

void registerOpenXlaTritonPases();

void registerOpenXlaTritonPipelines();

}  // namespace openxla::compiler::nvgpu::triton

#endif  // OPENXLA_COMPILER_NVGPU_DIALECT_TRITONFLOW_TRANSFORMS_PASSES_H_
