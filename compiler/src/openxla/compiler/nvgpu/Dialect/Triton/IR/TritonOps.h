// Copyright 2023 The OpenXLA Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef OPENXLA_COMPILER_NVGPU_DIALECT_TRITON_IR_TRITON_OPS_H
#define OPENXLA_COMPILER_NVGPU_DIALECT_TRITON_IR_TRITON_OPS_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/FunctionInterfaces.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define GET_OP_CLASSES
#include "openxla/compiler/nvgpu/Dialect/Triton/IR/TritonOps.h.inc"

#endif  // OPENXLA_COMPILER_NVGPU_DIALECT_TRITON_IR_TRITON_OPS_H
