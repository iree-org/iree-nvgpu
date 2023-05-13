// Copyright 2023 The OpenXLA Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "openxla/compiler/nvgpu/Dialect/Triton/IR/TritonDialect.h"

#include "openxla/compiler/nvgpu/Dialect/Triton/IR/TritonOps.h"

using namespace mlir;
using namespace openxla::compiler::nvgpu::triton;

#include "openxla/compiler/nvgpu/Dialect/Triton/IR/TritonDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// Triton dialect.
//===----------------------------------------------------------------------===//

void TritonDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "openxla/compiler/nvgpu/Dialect/Triton/IR/TritonOps.cpp.inc"
      >();
}
