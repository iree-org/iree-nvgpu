// Copyright 2023 The OpenXLA Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef OPENXLA_COMPILER_NVGPU_DIALECT_CUBLAS_IR_CUBLAS_OPS_H
#define OPENXLA_COMPILER_NVGPU_DIALECT_CUBLAS_IR_CUBLAS_OPS_H

#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "openxla/compiler/nvgpu/Dialect/CUBLAS/IR/CUBLASTypes.h"

#define GET_OP_CLASSES
#include "openxla/compiler/nvgpu/Dialect/CUBLAS/IR/CUBLASOps.h.inc"

#endif  // OPENXLA_COMPILER_NVGPU_DIALECT_CUBLAS_IR_CUBLAS_OPS_H
