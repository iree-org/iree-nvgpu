// Copyright 2023 The OpenXLA Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "openxla/compiler/nvgpu/Dialect/TritonFlow/IR/TritonFlowDialect.h"

#include "openxla/compiler/nvgpu/Dialect/TritonFlow/IR/TritonFlowOps.h"

using namespace mlir;
using namespace openxla::compiler::nvgpu::tritonflow;

#include "openxla/compiler/nvgpu/Dialect/TritonFlow/IR/TritonFlowDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// TritonFlow dialect.
//===----------------------------------------------------------------------===//

void TritonFlowDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "openxla/compiler/nvgpu/Dialect/TritonFlow/IR/TritonFlowOps.cpp.inc"
      >();
}
