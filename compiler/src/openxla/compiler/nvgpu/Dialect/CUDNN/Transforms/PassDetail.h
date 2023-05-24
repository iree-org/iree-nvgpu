// Copyright 2023 The OpenXLA Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef OPENXLA_COMPILER_NVGPU_DIALECT_CUDNN_TRANSFORMS_PASSDETAIL_H_
#define OPENXLA_COMPILER_NVGPU_DIALECT_CUDNN_TRANSFORMS_PASSDETAIL_H_

#include "compiler/src/iree/compiler/Dialect/Flow/IR/FlowDialect.h"
#include "compiler/src/iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "compiler/src/iree/compiler/Dialect/Util/IR/UtilDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "openxla/compiler/nvgpu/Dialect/CUDNN/IR/CUDNNDialect.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace openxla::compiler::nvgpu::cudnn {

// TODO: Deprecated. Remove this and use per-pass macros.
#define GEN_PASS_CLASSES
#include "openxla/compiler/nvgpu/Dialect/CUDNN/Transforms/Passes.h.inc"

}  // namespace openxla::compiler::nvgpu::cudnn

#endif  // OPENXLA_COMPILER_NVGPU_DIALECT_CUDNN_TRANSFORMS_PASSDETAIL_H_
