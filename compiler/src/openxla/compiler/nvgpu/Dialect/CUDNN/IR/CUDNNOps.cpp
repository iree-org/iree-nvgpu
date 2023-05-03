//===- CUDNNOps.cpp - CUDNN dialect ops -------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "openxla/compiler/nvgpu/Dialect/CUDNN/IR/CUDNNOps.h"

#include <cstdint>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/FunctionImplementation.h"
#include "mlir/IR/FunctionInterfaces.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Support/LogicalResult.h"
#include "openxla/compiler/nvgpu/Dialect/CUDNN/IR/CUDNNDialect.h"

namespace openxla::compiler::nvgpu::cudnn {

using namespace mlir;
using namespace mlir::iree_compiler;

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
  auto isCudnnTensor = [](Type type) {
    if (auto tensor = type.dyn_cast<CudnnTensorType>())
      return !tensor.isOpaque();
    return false;
  };

  // All arguments and results must be cuDNN tensors.
  if (!llvm::all_of(getArgumentTypes(), isCudnnTensor))
    return emitOpError("requires all arguments to be non-opaque cuDNN tensors");

  if (!llvm::all_of(getResultTypes(), isCudnnTensor))
    return emitOpError("requires all results to be non-opaque cuDNN tensors");

  // cuDNN graph must return exactly one result.
  if (getResultTypes().size() != 1)
    return emitOpError("requires exactly one cuDNN tensor result");

  return success();
}

//===----------------------------------------------------------------------===//
// cudnn.call operation
//===----------------------------------------------------------------------===//

using Diagnostic = std::function<InFlightDiagnostic()>;

// Get dimensions of the tensor argument after applying cuDNN tensor layout.
static SmallVector<int64_t> getTensorArgDims(CudnnTensorType tensor) {
  auto dims = tensor.getShape();

  std::optional<Layout> layout = tensor.getLayout();
  AffineMap strides = tensor.getStrides();

  // Handle one of the pre-defined cuDNN tensor layout.
  if (layout) {
    switch (*layout) {
      case Layout::KCHW:
      case Layout::NCHW:
        return {dims[0], dims[1], dims[2], dims[3]};
      case Layout::KHWC:
      case Layout::NHWC:
        return {dims[0], dims[2], dims[3], dims[1]};
    }
  }

  // Shuffle dimensions according to the strides permutation map.
  if (strides) {
    SmallVector<int64_t> shuffled(dims.size());
    for (unsigned d = 0; d < strides.getNumDims(); ++d)
      shuffled[d] = dims[strides.getDimPosition(d)];
    return shuffled;
  }

  // Simple row-major layout.
  return SmallVector<int64_t>(dims);
}

// Verifies that cuDNN graph type is compatible with tensor type.
static LogicalResult verifyTensorTypes(Diagnostic emitOpError,
                                       CudnnTensorType cudnnType,
                                       RankedTensorType tensorType,
                                       std::string_view kind,
                                       unsigned ordinal) {
  if (!cudnnType)
    return emitOpError() << "unsupported graph " << kind << " #" << ordinal;
  if (!tensorType)
    return emitOpError() << "unsupported " << kind << " #" << ordinal;

  if (cudnnType.getElementType() != tensorType.getElementType())
    return emitOpError() << kind << "#" << ordinal
                         << " graph tensor element type "
                         << cudnnType.getElementType()
                         << " does not match tensor element type "
                         << tensorType.getElementType();

  if (cudnnType.getShape().size() != tensorType.getShape().size())
    return emitOpError() << kind << " #" << ordinal << " graph tensor rank "
                         << cudnnType.getShape().size()
                         << " does not match tensor rank "
                         << tensorType.getRank();

  auto expectedShape = getTensorArgDims(cudnnType);
  if (expectedShape != tensorType.getShape())
    return emitOpError() << kind << " #" << ordinal << " shape "
                         << "[" << tensorType.getShape() << "]"
                         << " doesn't match the expected shape "
                         << "[" << expectedShape << "]";

  return success();
}

LogicalResult CallOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  auto g = symbolTable.lookupNearestSymbolFrom<GraphOp>(*this, getCalleeAttr());
  if (!g) return emitOpError() << "refers to an unknown cuDNN graph";

  auto graphArgs = g.getArgumentTypes();
  auto graphResults = g.getResultTypes();

  if (graphArgs.size() != getArguments().size())
    return emitOpError() << "has " << getArguments().size()
                         << "arguments, but the cuDNN graph expects "
                         << graphArgs.size() << " arguments";

  if (graphResults.size() != getResults().size())
    return emitOpError() << "has " << getResults().size()
                         << "results, but the cuDNN graph expects "
                         << graphResults.size() << " results";

  Diagnostic emitErr = [&]() { return emitOpError(); };

  for (unsigned i = 0; i < graphArgs.size(); ++i) {
    auto cudnnArg = graphArgs[i].dyn_cast<CudnnTensorType>();
    auto tensorArg = getArguments()[i].getType().dyn_cast<RankedTensorType>();
    if (failed(verifyTensorTypes(emitErr, cudnnArg, tensorArg, "argument", i)))
      return failure();
  }

  for (unsigned i = 0; i < graphResults.size(); ++i) {
    auto cudnnRet = graphResults[i].dyn_cast<CudnnTensorType>();
    auto tensorRet = getResults()[i].getType().dyn_cast<RankedTensorType>();
    if (failed(verifyTensorTypes(emitErr, cudnnRet, tensorRet, "result", i)))
      return failure();
  }

  return success();
}

}  // namespace openxla::compiler::nvgpu::cudnn

#define GET_OP_CLASSES
#include "openxla/compiler/nvgpu/Dialect/CUDNN/IR/CUDNNOps.cpp.inc"
