//===- CUDNNOps.cpp - CUDNN dialect ops -------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "openxla/compiler/nvgpu/Dialect/CUDNN/IR/CUDNNOps.h"

#include "llvm/ADT/STLExtras.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/FunctionImplementation.h"
#include "mlir/IR/FunctionInterfaces.h"
#include "mlir/IR/OpImplementation.h"
#include "openxla/compiler/nvgpu/Dialect/CUDNN/IR/CUDNNDialect.h"

namespace openxla::compiler::nvgpu::cudnn {

using namespace mlir;

//===----------------------------------------------------------------------===//
// cudnn.graph operation
//===----------------------------------------------------------------------===//

ParseResult GraphOp::parse(OpAsmParser &parser, OperationState &result) {
  auto buildFuncType =
      [](Builder &builder, ArrayRef<Type> argTypes, ArrayRef<Type> results,
         function_interface_impl::VariadicFlag,
         std::string &) { return builder.getFunctionType(argTypes, results); };
  return function_interface_impl::parseFunctionOp(
      parser, result, /*allowVariadic=*/false,
      getFunctionTypeAttrName(result.name), buildFuncType,
      getArgAttrsAttrName(result.name), getResAttrsAttrName(result.name));
}

void GraphOp::print(OpAsmPrinter &p) {
  function_interface_impl::printFunctionOp(
      p, *this, /*isVariadic=*/false, getFunctionTypeAttrName(),
      getArgAttrsAttrName(), getResAttrsAttrName());
}

LogicalResult GraphOp::verifyType() {
  // Check if type is a legal cuDNN tensor type.
  auto is_cudnn_tensor = [](Type type) {
    if (auto tensor = type.dyn_cast<cudnn::TensorType>())
      return !tensor.isOpaque();
    return false;
  };

  // All arguments and results must be cuDNN tensors.
  if (!llvm::all_of(getArgumentTypes(), is_cudnn_tensor))
    return emitOpError("requires all arguments to be non-opaque cuDNN tensors");

  if (!llvm::all_of(getResultTypes(), is_cudnn_tensor))
    return emitOpError("requires all results to be non-opaque cuDNN tensors");

  // cuDNN graph must return exactly one result.
  if (getResultTypes().size() != 1)
    return emitOpError("requires exactly one cuDNN tensor result");

  return success();
}

}  // namespace openxla::compiler::nvgpu::cudnn

#define GET_OP_CLASSES
#include "openxla/compiler/nvgpu/Dialect/CUDNN/IR/CUDNNOps.cpp.inc"
