// Copyright 2023 The OpenXLA Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "openxla/compiler/nvgpu/Dialect/CUBLAS/IR/CUBLASDialect.h"

#include "openxla/compiler/nvgpu/Dialect/CUBLAS/IR/CUBLASOps.h"

using namespace mlir;
using namespace openxla::compiler::nvgpu::cublas;

#include "openxla/compiler/nvgpu/Dialect/CUBLAS/IR/CUBLASDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// CUBLAS dialect.
//===----------------------------------------------------------------------===//

void CUBLASDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "openxla/compiler/nvgpu/Dialect/CUBLAS/IR/CUBLASOps.cpp.inc"
      >();
  registerAttrs();
  registerTypes();
}
