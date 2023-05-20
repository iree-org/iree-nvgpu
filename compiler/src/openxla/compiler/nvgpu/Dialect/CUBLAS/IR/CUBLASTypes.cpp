// Copyright 2023 The OpenXLA Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "openxla/compiler/nvgpu/Dialect/CUBLAS/IR/CUBLASTypes.h"

#include "llvm/ADT/TypeSwitch.h"
#include "mlir/IR/Builders.h"
#include "openxla/compiler/nvgpu/Dialect/CUBLAS/IR/CUBLASDialect.h"

using namespace mlir;
using namespace openxla::compiler::nvgpu::cublas;

#include "openxla/compiler/nvgpu/Dialect/CUBLAS/IR/CUBLASEnums.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "openxla/compiler/nvgpu/Dialect/CUBLAS/IR/CUBLASAttrs.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "openxla/compiler/nvgpu/Dialect/CUBLAS/IR/CUBLASTypes.cpp.inc"

void CUBLASDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "openxla/compiler/nvgpu/Dialect/CUBLAS/IR/CUBLASTypes.cpp.inc"
      >();
}

void CUBLASDialect::registerAttrs() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "openxla/compiler/nvgpu/Dialect/CUBLAS/IR/CUBLASAttrs.cpp.inc"
      >();
}
