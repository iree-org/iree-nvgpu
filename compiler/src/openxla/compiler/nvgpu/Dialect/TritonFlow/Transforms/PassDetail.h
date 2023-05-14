// Copyright 2023 The OpenXLA Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef OPENXLA_COMPILER_NVGPU_DIALECT_TRITONFLOW_TRANSFORMS_PASSDETAIL_H_
#define OPENXLA_COMPILER_NVGPU_DIALECT_TRITONFLOW_TRANSFORMS_PASSDETAIL_H_

#include <compiler/src/iree/compiler/Dialect/Flow/IR/FlowDialect.h>
#include <compiler/src/iree/compiler/Dialect/HAL/IR/HALDialect.h>

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "openxla/compiler/nvgpu/Dialect/TritonFlow/IR/TritonFlowDialect.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

namespace openxla::compiler::nvgpu::tritonflow {

#define GEN_PASS_CLASSES
#include "openxla/compiler/nvgpu/Dialect/TritonFlow/Transforms/Passes.h.inc"

}  // namespace openxla::compiler::nvgpu::tritonflow

#endif  // OPENXLA_COMPILER_NVGPU_DIALECT_TRITONFLOW_TRANSFORMS_PASSDETAIL_H_
