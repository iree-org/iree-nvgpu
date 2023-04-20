// Copyright 2023 The OpenXLA Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "openxla/compiler/nvgpu/Dialect/CUDNN/Conversion/ConvertCUDNNToRuntime.h"

#include <cstdint>
#include <memory>
#include <string>

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
  func::FuncOp getTensorCreateFunction(ModuleOp module, int64_t rank);

  // Imports `@cudnn.operation_graph.create` into the module.
  func::FuncOp getOperationGraphCreateFunction(ModuleOp module);

 private:
  func::FuncOp addDecl(ModuleOp module, StringAttr name,
                       FunctionType function_type);

  SymbolTable &symTable(ModuleOp module);

  SymbolTableCollection symTable_;
};

SymbolTable &CudnnAPI::symTable(ModuleOp module) {
  return symTable_.getSymbolTable(module);
}

func::FuncOp CudnnAPI::addDecl(ModuleOp module, StringAttr name,
                               FunctionType function_type) {
  if (auto fn = symTable_.lookupNearestSymbolFrom<func::FuncOp>(module, name))
    return fn;

  auto b = ImplicitLocOpBuilder::atBlockEnd(
      UnknownLoc::get(module->getContext()), module.getBody());
  auto fn = b.create<func::FuncOp>(name, function_type);
  fn.setPrivate();
  symTable(module).insert(fn);
  return fn;
}

func::FuncOp CudnnAPI::getTensorCreateFunction(ModuleOp module, int64_t rank) {
  MLIRContext *ctx = module->getContext();
  SmallVector<Type> args(/*dtype*/ 1 + rank, IntegerType::get(ctx, 64));
  SmallVector<Type> rets = {cudnn::TensorType::get(ctx)};
  auto function_type = FunctionType::get(ctx, args, rets);
  auto function_name =
      StringAttr::get(ctx, llvm::formatv("cudnn.tensor.create.{0}d", rank));
  return addDecl(module, function_name, function_type);
}

func::FuncOp CudnnAPI::getOperationGraphCreateFunction(ModuleOp module) {
  MLIRContext *ctx = module->getContext();
  SmallVector<Type> args = {cudnn::TensorType::get(ctx)};
  SmallVector<Type> rets = {cudnn::OperationGraphType::get(ctx)};
  auto function_type = FunctionType::get(ctx, args, rets);
  auto function_name = StringAttr::get(ctx, "cudnn.operation_graph.create");
  return addDecl(module, function_name, function_type);
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

      // TODO(ezhulenev): Support tensors with non standard format.
      assert(!tensorArg.getLayout().has_value() && !tensorArg.getStrides());

      // TODO(ezhulenev): Get a valid cuDNN data type from the tensor element
      // type. Currently we hardcode `0` (CUDNN_DATA_FLOAT).
      SmallVector<Value> args = {b.create<arith::ConstantIntOp>(0, 64)};
      for (int64_t dim : shape) {
        assert(dim >= 0 && "dynamic shapes are not supported");
        args.push_back(b.create<arith::ConstantIntOp>(dim, 64));
      }

      auto createTensor = api->getTensorCreateFunction(module, shape.size());
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
    auto createOpGraph = api->getOperationGraphCreateFunction(module);
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
