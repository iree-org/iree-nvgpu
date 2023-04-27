//===- CUDNNDialect.h - CUDNN dialect ---------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CUDNN_CUDNNDIALECT_H
#define CUDNN_CUDNNDIALECT_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"

// Must be included after all MLIR headers.
#include "openxla/compiler/nvgpu/Dialect/CUDNN/IR/CUDNNDialect.h.inc"

#endif  // CUDNN_CUDNNDIALECT_H
