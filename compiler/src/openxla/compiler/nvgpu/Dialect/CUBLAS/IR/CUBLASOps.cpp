// Copyright 2023 The OpenXLA Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "openxla/compiler/nvgpu/Dialect/CUBLAS/IR/CUBLASOps.h"

#include "openxla/compiler/nvgpu/Dialect/CUBLAS/IR/CUBLASDialect.h"

using namespace mlir;
using namespace mlir::iree_compiler;

#define GET_OP_CLASSES
#include "openxla/compiler/nvgpu/Dialect/CUBLAS/IR/CUBLASOps.cpp.inc"
