// Copyright 2023 The OpenXLA Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "openxla/compiler/nvgpu/Dialect/CUBLAS/IR/CUBLASOps.h"

#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "openxla/compiler/nvgpu/Dialect/CUBLAS/IR/CUBLASDialect.h"

using namespace mlir;
using namespace mlir::iree_compiler;

namespace openxla::compiler::nvgpu::cublas {

// Verifies that |dynamicDims| contains the appropriate number of dims for all
// of the dynamic dimensions in |values|.
static LogicalResult verifyOpDynamicDims(Operation *op, ValueRange values,
                                         ValueRange dynamicDims) {
  unsigned requiredCount = 0;
  for (auto value : values) {
    if (auto shapedType = value.getType().dyn_cast<ShapedType>()) {
      requiredCount += shapedType.getNumDynamicDims();
    }
  }
  if (dynamicDims.size() != requiredCount) {
    return op->emitOpError()
           << "value set has " << requiredCount
           << " dynamic dimensions but only " << dynamicDims.size()
           << " dimension values are attached";
  }
  return success();
}

//===----------------------------------------------------------------------===//
// cublas.gemm operation
//===----------------------------------------------------------------------===//

ValueRange GemmOp::getOperandDynamicDims(unsigned idx) {
  return IREE::Util::findVariadicDynamicDims(idx, getArguments(),
                                             getArgumentDims());
}
ValueRange GemmOp::getResultDynamicDims(unsigned idx) {
  return IREE::Util::findVariadicDynamicDims(idx, getResults(),
                                             getResultDims());
}

LogicalResult GemmOp::verify() {
  Operation *op = getOperation();
  if (failed(verifyOpDynamicDims(op, getArguments(), getArgumentDims())) ||
      failed(verifyOpDynamicDims(op, getResults(), getResultDims()))) {
    return failure();
  }

  // Gemm takes `A` and `B` tensor arguments, and optional `C` tensor argument.
  if (getNumOperands() < 2 || getNumOperands() > 3) {
    return emitOpError() << "must have two or three arguments";
  }

  // Gemm always produces a single result.
  if (getNumResults() != 1) {
    return emitOpError() << "must have exactly one result";
  }

  // Only the `C` tensor argument can used as a tied operand.
  if (auto tiedOperand = getTiedResultOperandIndex(0);
      tiedOperand.has_value() && *tiedOperand != 2) {
    return emitOpError() << "only third argument can be used as a tied operand";
  }

  return success();
}

}  // namespace openxla::compiler::nvgpu::cublas

#define GET_OP_CLASSES
#include "openxla/compiler/nvgpu/Dialect/CUBLAS/IR/CUBLASOps.cpp.inc"
