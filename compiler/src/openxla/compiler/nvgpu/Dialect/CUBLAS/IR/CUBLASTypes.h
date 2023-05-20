// Copyright 2023 The OpenXLA Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef OPENXLA_COMPILER_NVGPU_DIALECT_CUBLAS_IR_CUBLAS_TYPES_H
#define CUBLAS_CUBLASTYPES_H

#include "mlir/IR/BuiltinTypes.h"

// Must be included after all MLIR headers.
#include "openxla/compiler/nvgpu/Dialect/CUBLAS/IR/CUBLASEnums.h.inc"

#define GET_ATTRDEF_CLASSES
#include "openxla/compiler/nvgpu/Dialect/CUBLAS/IR/CUBLASAttrs.h.inc"

#define GET_TYPEDEF_CLASSES
#include "openxla/compiler/nvgpu/Dialect/CUBLAS/IR/CUBLASTypes.h.inc"

#endif  // OPENXLA_COMPILER_NVGPU_DIALECT_CUBLAS_IR_CUBLAS_TYPES_H
