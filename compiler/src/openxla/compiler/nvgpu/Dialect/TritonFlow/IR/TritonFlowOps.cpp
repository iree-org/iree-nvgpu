// Copyright 2023 The OpenXLA Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "openxla/compiler/nvgpu/Dialect/TritonFlow/IR/TritonFlowOps.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/FunctionImplementation.h"
#include "mlir/IR/FunctionInterfaces.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Support/LogicalResult.h"
#include "openxla/compiler/nvgpu/Dialect/TritonFlow/IR/TritonFlowDialect.h"

namespace openxla::compiler::nvgpu::triton {

using namespace mlir;

//===----------------------------------------------------------------------===//
// triton.dispatch operation
//===----------------------------------------------------------------------===//

LogicalResult DispatchOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  auto fn = symbolTable.lookupNearestSymbolFrom(*this, getCalleeAttr());
  if (!fn) return emitOpError() << "refers to an unknown Triton callee";
  return success();
}

}  // namespace openxla::compiler::nvgpu::triton

#define GET_OP_CLASSES
#include "openxla/compiler/nvgpu/Dialect/TritonFlow/IR/TritonFlowOps.cpp.inc"
