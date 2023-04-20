// Copyright 2023 The OpenXLA Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "openxla/compiler/nvgpu/Dialect/CUDNN/Conversion/ConvertCUDNNToRuntime.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Transforms/DialectConversion.h"
#include "openxla/compiler/nvgpu/Dialect/CUDNN/IR/CUDNNOps.h"
#include "openxla/compiler/nvgpu/Dialect/CUDNN/IR/CUDNNTypes.h"

namespace openxla::compiler::nvgpu::cudnn {

using namespace mlir;

namespace {

//===----------------------------------------------------------------------===//
// CudnnAPI for importing cuDNN VM module functions
//===----------------------------------------------------------------------===//

class CudnnAPI {
 public:
  // Imports `@cudnn.tensor.create` into the module.
  func::FuncOp getTensorCreateFunction(PatternRewriter &rewriter,
                                       ModuleOp module, int64_t rank,
                                       std::optional<Layout> layout);

  // Imports `@cudnn.operation_graph.create` into the module.
  func::FuncOp getOperationGraphCreateFunction(PatternRewriter &rewriter,
                                               ModuleOp module);

 private:
  func::FuncOp addDecl(PatternRewriter &rewriter, ModuleOp module,
                       StringAttr name, FunctionType function_type);

  SymbolTable &symTable(ModuleOp module);

  SymbolTableCollection symTable_;
};

SymbolTable &CudnnAPI::symTable(ModuleOp module) {
  return symTable_.getSymbolTable(module);
}

func::FuncOp CudnnAPI::addDecl(PatternRewriter &rewriter, ModuleOp module,
                               StringAttr name, FunctionType function_type) {
  if (auto fn = symTable_.lookupNearestSymbolFrom<func::FuncOp>(module, name))
    return fn;

  ImplicitLocOpBuilder b(UnknownLoc::get(module->getContext()), rewriter);
  b.setInsertionPointToEnd(module.getBody());

  auto fn = b.create<func::FuncOp>(name, function_type);
  fn.setPrivate();
  symTable(module).insert(fn);
  return fn;
}

func::FuncOp CudnnAPI::getTensorCreateFunction(PatternRewriter &rewriter,
                                               ModuleOp module, int64_t rank,
                                               std::optional<Layout> layout) {
  MLIRContext *ctx = module->getContext();
  SmallVector<Type> args(/*dtype*/ 1 + rank, IntegerType::get(ctx, 64));
  SmallVector<Type> rets = {cudnn::TensorType::get(ctx)};
  auto function_type = FunctionType::get(ctx, args, rets);

  std::string function_name = llvm::formatv("cudnn.tensor.create.{0}d", rank);
  if (layout) function_name += "." + stringifyLayout(*layout).lower();

  return addDecl(rewriter, module, StringAttr::get(ctx, function_name),
                 function_type);
}

func::FuncOp CudnnAPI::getOperationGraphCreateFunction(
    PatternRewriter &rewriter, ModuleOp module) {
  MLIRContext *ctx = module->getContext();
  SmallVector<Type> args = {cudnn::TensorType::get(ctx)};
  SmallVector<Type> rets = {cudnn::OperationGraphType::get(ctx)};
  auto function_type = FunctionType::get(ctx, args, rets);
  auto function_name = StringAttr::get(ctx, "cudnn.operation_graph.create");
  return addDecl(rewriter, module, function_name, function_type);
}

//===----------------------------------------------------------------------===//
// Base class for all cuDNN op conversions
//===----------------------------------------------------------------------===//

template <typename T>
struct CudnnOpConversionPattern : public OpConversionPattern<T> {
  CudnnOpConversionPattern(TypeConverter &typeConverter, MLIRContext *ctx,
                           std::shared_ptr<CudnnAPI> api)
      : OpConversionPattern<T>(typeConverter, ctx), api(std::move(api)) {}

  std::shared_ptr<CudnnAPI> api;
};

//===----------------------------------------------------------------------===//
// Helper functions for converting to runtime API calls
//===----------------------------------------------------------------------===//

static FailureOr<DataType> getDataType(Type elementType) {
  if (elementType.isF32()) return DataType::FLOAT;
  if (elementType.isF64()) return DataType::DOUBLE;
  if (elementType.isF16()) return DataType::HALF;
  if (elementType.isBF16()) return DataType::BFLOAT16;
  if (elementType.isInteger(1)) return DataType::BOOLEAN;
  if (elementType.isInteger(8)) return DataType::INT8;
  if (elementType.isInteger(32)) return DataType::INT32;
  if (elementType.isInteger(64)) return DataType::INT64;
  if (elementType.isUnsignedInteger(8)) return DataType::UINT8;
  return failure();
}

//===----------------------------------------------------------------------===//
// cudnn.graph
//===----------------------------------------------------------------------===//

// Converts `cudnn.graph` operations to a function building cuDNN operation
// graph using cuDNN runtime APIs.
struct ConvertCudnnGraphOp : public CudnnOpConversionPattern<cudnn::GraphOp> {
  using CudnnOpConversionPattern::CudnnOpConversionPattern;

  LogicalResult matchAndRewrite(
      cudnn::GraphOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    MLIRContext *ctx = rewriter.getContext();
    ImplicitLocOpBuilder b(op->getLoc(), rewriter);

    ModuleOp module = op->getParentOfType<ModuleOp>();

    // Create a builder function right before the `cudnn.graph` operation.
    b.setInsertionPoint(op);
    auto builder = b.create<func::FuncOp>(
        StringAttr::get(ctx, llvm::formatv("{0}.builder", op.getName())),
        FunctionType::get(ctx, {}, {OperationGraphType::get(ctx)}));

    Block *body = builder.addEntryBlock();
    b.setInsertionPointToStart(body);

    // Mapping from cudnn.graph arguments to values in the builder body.
    SmallVector<Value> mappedArgs;

    // Create cuDNN tensor for every graph argument.
    for (auto arg : op.getArgumentTypes()) {
      auto tensorArg = arg.cast<cudnn::TensorType>();
      auto shape = tensorArg.getShape();

      if (llvm::any_of(shape, [](int64_t dim) { return dim < 0; }))
        return rewriter.notifyMatchFailure(
            op, "dynamic dimensions are not supported");

      if (tensorArg.getStrides())
        return rewriter.notifyMatchFailure(op,
                                           "strided layout is not supported");

      auto dtype = getDataType(tensorArg.getElementType());
      if (failed(dtype))
        return rewriter.notifyMatchFailure(op, "unsupported element type");

      SmallVector<Value> args = {
          b.create<arith::ConstantIntOp>(static_cast<int64_t>(*dtype), 64)};
      for (int64_t dim : shape)
        args.push_back(b.create<arith::ConstantIntOp>(dim, 64));

      auto createTensor = api->getTensorCreateFunction(
          rewriter, module, shape.size(), tensorArg.getLayout());
      auto tensor = b.create<func::CallOp>(createTensor.getSymName(),
                                           cudnn::TensorType::get(ctx), args);
      mappedArgs.push_back(tensor.getResult(0));
    }

    // Move cudnn.graph body into the builder function.
    rewriter.inlineBlockBefore(&op.getBody().front(), body, body->end(),
                               mappedArgs);

    rewriter.eraseOp(op);

    return success();
  }
};

//===----------------------------------------------------------------------===//
// cudnn.return
//===----------------------------------------------------------------------===//

struct ConvertCudnnReturnOp : public CudnnOpConversionPattern<cudnn::ReturnOp> {
  using CudnnOpConversionPattern::CudnnOpConversionPattern;

  LogicalResult matchAndRewrite(
      cudnn::ReturnOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    MLIRContext *ctx = rewriter.getContext();
    ImplicitLocOpBuilder b(op->getLoc(), rewriter);

    ModuleOp module = op->getParentOfType<ModuleOp>();

    // Create an operation graph from the returned tensor results.
    auto createOpGraph = api->getOperationGraphCreateFunction(rewriter, module);
    auto opGraph = b.create<func::CallOp>(createOpGraph.getSymName(),
                                          OperationGraphType::get(ctx),
                                          adaptor.getOperands());
    b.create<func::ReturnOp>(opGraph->getResults());

    rewriter.eraseOp(op);
    return success();
  }
};

}  // namespace

void populateCudnnToRuntimePatterns(mlir::TypeConverter &typeConverter,
                                    mlir::RewritePatternSet &patterns) {
  MLIRContext *ctx = patterns.getContext();
  auto api = std::make_shared<CudnnAPI>();
  patterns.insert<ConvertCudnnGraphOp>(typeConverter, ctx, api);
  patterns.insert<ConvertCudnnReturnOp>(typeConverter, ctx, api);
}

}  // namespace openxla::compiler::nvgpu::cudnn
