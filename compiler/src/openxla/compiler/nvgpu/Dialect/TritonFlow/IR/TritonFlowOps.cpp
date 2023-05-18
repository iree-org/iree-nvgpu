// Copyright 2023 The OpenXLA Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "openxla/compiler/nvgpu/Dialect/TritonFlow/IR/TritonFlowOps.h"

#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
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

namespace openxla::compiler::nvgpu::tritonflow {

using namespace mlir;
using namespace mlir::iree_compiler;

//===----------------------------------------------------------------------===//
// triton.executable operation
//===----------------------------------------------------------------------===//

LogicalResult ExecutableOp::verify() {
  auto innerModules = getBlock().getOps<ModuleOp>();

  if (llvm::count_if(innerModules, [](ModuleOp) { return true; }) != 1)
    return emitOpError()
           << "expected exactly one inner builtin.module operation";

  return success();
}

//===----------------------------------------------------------------------===//
// triton.executable.export operation
//===----------------------------------------------------------------------===//

LogicalResult ExecutableExportOp::verifySymbolUses(
    SymbolTableCollection &symbolTable) {
  auto innerModule = getParentOp<ExecutableOp>().getInnerModule();
  if (!symbolTable.lookupNearestSymbolFrom(innerModule, getFunctionRefAttr())) {
    return emitOpError() << "refers to an unknown Triton function";
  }
  return success();
}

//===----------------------------------------------------------------------===//
// triton.call operation
//===----------------------------------------------------------------------===//

LogicalResult CallOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  if (!symbolTable.lookupNearestSymbolFrom(*this, getCalleeAttr()))
    return emitOpError() << "refers to an unknown Triton callee";
  return success();
}

}  // namespace openxla::compiler::nvgpu::tritonflow

#define GET_OP_CLASSES
#include "openxla/compiler/nvgpu/Dialect/TritonFlow/IR/TritonFlowOps.cpp.inc"
