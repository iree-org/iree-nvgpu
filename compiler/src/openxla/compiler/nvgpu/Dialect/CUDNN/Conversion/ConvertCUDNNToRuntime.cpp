// Copyright 2023 The OpenXLA Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "openxla/compiler/nvgpu/Dialect/CUDNN/Conversion/ConvertCUDNNToRuntime.h"

#include <iree/compiler/Dialect/Util/IR/UtilOps.h>
#include <iree/compiler/Dialect/Util/IR/UtilTypes.h>

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>

#include "compiler/src/iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "compiler/src/iree/compiler/Dialect/HAL/IR/HALTypes.h"
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
#include "mlir/IR/Value.h"
#include "mlir/Transforms/DialectConversion.h"
#include "openxla/compiler/nvgpu/Dialect/CUDNN/IR/CUDNNOps.h"
#include "openxla/compiler/nvgpu/Dialect/CUDNN/IR/CUDNNTypes.h"

namespace openxla::compiler::nvgpu::cudnn {

namespace IREE = mlir::iree_compiler::IREE;

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

  // Imports pointwise unary `@cudnn.{op}` into the module.
  func::FuncOp getPointwiseUnaryFunction(PatternRewriter &rewriter,
                                         ModuleOp module, std::string_view op);

  // Imports pointwise binary `@cudnn.{op}` into the module.
  func::FuncOp getPointwiseBinaryFunction(PatternRewriter &rewriter,
                                          ModuleOp module, std::string_view op);

  // Imports `@cudnn.bias` into the module.
  func::FuncOp getBiasFunction(PatternRewriter &rewriter, ModuleOp module);

  // Imports `@cudnn.convolution` into the module.
  func::FuncOp getConvolutionFunction(PatternRewriter &rewriter,
                                      ModuleOp module, int64_t spatial_dims);

  // Imports `@cudnn.handle` into the module.
  func::FuncOp getHandleFunction(PatternRewriter &rewriter, ModuleOp module);

  // Imports `@cudnn.operation_graph.create` into the module.
  func::FuncOp getOperationGraphCreateFunction(PatternRewriter &rewriter,
                                               ModuleOp module);

  // Imports `@cudnn.executable.create` into the module.
  func::FuncOp getExecutableCreateFunction(PatternRewriter &rewriter,
                                           ModuleOp module);

  // Imports `@cudnn.execute` into the module.
  func::FuncOp getExecuteFunction(PatternRewriter &rewriter, ModuleOp module,
                                  int64_t num_args);

  SymbolTable &symTable(ModuleOp module);

 private:
  func::FuncOp addDecl(PatternRewriter &rewriter, ModuleOp module,
                       StringAttr name, FunctionType function_type);

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
  SmallVector<Type> rets = {CudnnTensorType::get(ctx)};
  auto functionType = FunctionType::get(ctx, args, rets);

  std::string functionName = llvm::formatv("cudnn.tensor.create.{0}d", rank);
  if (layout) functionName += "." + stringifyLayout(*layout).lower();

  return addDecl(rewriter, module, StringAttr::get(ctx, functionName),
                 functionType);
}

func::FuncOp CudnnAPI::getPointwiseUnaryFunction(PatternRewriter &rewriter,
                                                 ModuleOp module,
                                                 std::string_view op) {
  MLIRContext *ctx = module->getContext();
  auto tensor = CudnnTensorType::get(ctx);
  auto f32 = Float32Type::get(ctx);
  auto i32 = IntegerType::get(ctx, 32);

  SmallVector<Type> args = {/*x=*/tensor, /*alpha=*/f32, /*is_virtual=*/i32};
  SmallVector<Type> rets = {/*y=*/tensor};
  auto functionType = FunctionType::get(ctx, args, rets);

  return addDecl(rewriter, module, StringAttr::get(ctx, Twine("cudnn.") + op),
                 functionType);
}

func::FuncOp CudnnAPI::getPointwiseBinaryFunction(PatternRewriter &rewriter,
                                                  ModuleOp module,
                                                  std::string_view op) {
  MLIRContext *ctx = module->getContext();
  auto tensor = CudnnTensorType::get(ctx);
  auto f32 = Float32Type::get(ctx);
  auto i32 = IntegerType::get(ctx, 32);

  SmallVector<Type> args = {/*x=*/tensor, /*alpha=*/f32, /*b=*/tensor,
                            /*alpha2=*/f32, /*is_virtual=*/i32};
  SmallVector<Type> rets = {/*y=*/tensor};
  auto functionType = FunctionType::get(ctx, args, rets);

  return addDecl(rewriter, module, StringAttr::get(ctx, Twine("cudnn.") + op),
                 functionType);
}

func::FuncOp CudnnAPI::getBiasFunction(PatternRewriter &rewriter,
                                       ModuleOp module) {
  MLIRContext *ctx = module->getContext();
  auto tensor = CudnnTensorType::get(ctx);
  auto i32 = IntegerType::get(ctx, 32);

  SmallVector<Type> args = {/*x=*/tensor, /*b=*/tensor, /*is_virtual=*/i32};
  SmallVector<Type> rets = {/*y=*/tensor};
  auto functionType = FunctionType::get(ctx, args, rets);

  return addDecl(rewriter, module, StringAttr::get(ctx, "cudnn.bias"),
                 functionType);
}

func::FuncOp CudnnAPI::getConvolutionFunction(PatternRewriter &rewriter,
                                              ModuleOp module,
                                              int64_t spatial_dims) {
  MLIRContext *ctx = module->getContext();
  auto tensor = CudnnTensorType::get(ctx);
  auto i32 = IntegerType::get(ctx, 32);
  auto i64 = IntegerType::get(ctx, 64);

  SmallVector<Type> args = {/*x=*/tensor, /*w=*/tensor};
  args.append(spatial_dims, /*stride=*/i64);
  args.append(spatial_dims, /*pre_padding=*/i64);
  args.append(spatial_dims, /*post_patting=*/i64);
  args.append(spatial_dims, /*dilation=*/i64);
  args.append(1, /*is_virtual=*/i32);
  args.append(1, /*mode=*/i32);

  SmallVector<Type> rets = {/*y=*/tensor};
  auto functionType = FunctionType::get(ctx, args, rets);

  auto functionName = llvm::formatv("cudnn.convolution.{0}d", spatial_dims);
  return addDecl(rewriter, module, StringAttr::get(ctx, functionName),
                 functionType);
}

func::FuncOp CudnnAPI::getHandleFunction(PatternRewriter &rewriter,
                                         ModuleOp module) {
  MLIRContext *ctx = module->getContext();
  SmallVector<Type> args = {IREE::HAL::DeviceType::get(ctx)};
  SmallVector<Type> rets = {HandleType::get(ctx)};
  auto functionType = FunctionType::get(ctx, args, rets);
  auto functionName = StringAttr::get(ctx, "cudnn.handle");
  return addDecl(rewriter, module, functionName, functionType);
}

func::FuncOp CudnnAPI::getOperationGraphCreateFunction(
    PatternRewriter &rewriter, ModuleOp module) {
  MLIRContext *ctx = module->getContext();
  SmallVector<Type> args = {HandleType::get(ctx), CudnnTensorType::get(ctx)};
  SmallVector<Type> rets = {cudnn::OperationGraphType::get(ctx)};
  auto functionType = FunctionType::get(ctx, args, rets);
  auto functionName = StringAttr::get(ctx, "cudnn.operation_graph.create");
  return addDecl(rewriter, module, functionName, functionType);
}

func::FuncOp CudnnAPI::getExecutableCreateFunction(PatternRewriter &rewriter,
                                                   ModuleOp module) {
  MLIRContext *ctx = module->getContext();
  SmallVector<Type> args = {OperationGraphType::get(ctx)};
  SmallVector<Type> rets = {ExecutableType::get(ctx)};
  return addDecl(rewriter, module,
                 StringAttr::get(ctx, "cudnn.executable.create"),
                 FunctionType::get(ctx, args, rets));
}

func::FuncOp CudnnAPI::getExecuteFunction(PatternRewriter &rewriter,
                                          ModuleOp module, int64_t num_args) {
  MLIRContext *ctx = module->getContext();

  SmallVector<Type> args = {ExecutableType::get(ctx)};
  args.resize(args.size() + num_args, IREE::HAL::BufferViewType::get(ctx));

  SmallVector<Type> rets = {IREE::HAL::BufferViewType::get(ctx)};
  auto functionType = FunctionType::get(ctx, args, rets);

  auto functionName = llvm::formatv("cudnn.execute.{0}", num_args);
  return addDecl(rewriter, module, StringAttr::get(ctx, functionName),
                 functionType);
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

// Returns true if value is an intermediate cuDNN tensor (virtual tensor).
static bool IsVirtual(TypedValue<CudnnTensorType> tensor) {
  return llvm::none_of(tensor.getUsers(),
                       [](Operation *op) { return isa<cudnn::ReturnOp>(op); });
}

//===----------------------------------------------------------------------===//
// cudnn.handle
//===----------------------------------------------------------------------===//

struct ConvertCudnnHandleOp : public CudnnOpConversionPattern<HandleOp> {
  using CudnnOpConversionPattern::CudnnOpConversionPattern;

  LogicalResult matchAndRewrite(
      HandleOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    MLIRContext *ctx = rewriter.getContext();

    auto fn =
        this->api->getHandleFunction(rewriter, op->getParentOfType<ModuleOp>());
    rewriter.replaceOpWithNewOp<func::CallOp>(
        op, fn.getSymName(), HandleType::get(ctx), adaptor.getOperands());
    return success();
  }
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
        FunctionType::get(ctx, {HandleType::get(ctx)},
                          {OperationGraphType::get(ctx)}));
    api->symTable(module).insert(builder);

    Block *body = builder.addEntryBlock();
    b.setInsertionPointToStart(body);

    // Mapping from cudnn.graph arguments to values in the builder body.
    SmallVector<Value> mappedArgs;

    // Create cuDNN tensor for every graph argument.
    for (auto arg : op.getArgumentTypes()) {
      auto tensorArg = arg.cast<CudnnTensorType>();
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
                                           CudnnTensorType::get(ctx), args);
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
// cudnn.call
//===----------------------------------------------------------------------===//

struct ConvertCudnnCallOp : public CudnnOpConversionPattern<cudnn::CallOp> {
  using CudnnOpConversionPattern::CudnnOpConversionPattern;

  LogicalResult matchAndRewrite(
      cudnn::CallOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    MLIRContext *ctx = rewriter.getContext();
    ImplicitLocOpBuilder b(op->getLoc(), rewriter);

    ModuleOp module = op->getParentOfType<ModuleOp>();

    // Find the graph builder from the callee graph name.
    auto builder = api->symTable(module).lookup<func::FuncOp>(
        (op.getCallee() + ".builder").str());
    if (!builder)
      return rewriter.notifyMatchFailure(op, "graph builder was not found");

    // If cuDNN handle is loaded from a global, we can construct executable at
    // module initialization time, otherwise construct it at the call site.
    auto loadHandle = dyn_cast_or_null<IREE::Util::GlobalLoadOp>(
        op.getHandle().getDefiningOp());
    Value executable =
        loadHandle
            ? getGlobalExecutable(op, b, rewriter, builder, loadHandle)
            : getLocalExecutable(op, b, rewriter, builder, op.getHandle());

    auto bufferView = IREE::HAL::BufferViewType::get(ctx);

    // Export all tensor arguments to HAL buffer views.
    SmallVector<Value> args = {executable};
    for (auto [index, arg] : llvm::enumerate(op.getArguments())) {
      auto name = llvm::formatv("{0}.arg.{1}", op.getCallee(), index);
      args.push_back(b.create<IREE::HAL::TensorExportOp>(
          bufferView, arg, TypeAttr::get(arg.getType()),
          StringAttr::get(ctx, name)));
    }

    // Call execute function with executable and buffer arguments.
    auto execute =
        api->getExecuteFunction(rewriter, module, op.getArguments().size());
    auto executed =
        b.create<func::CallOp>(execute.getSymName(), bufferView, args);

    // Import HAL buffers view back as tensors.
    Type resultTy = op->getResult(0).getType();
    rewriter.replaceOpWithNewOp<IREE::HAL::TensorImportOp>(
        op, resultTy, executed->getResult(0), TypeAttr::get(resultTy),
        StringAttr::get(ctx, op.getCallee() + ".result"));

    return success();
  }

 private:
  Value getLocalExecutable(cudnn::CallOp op, ImplicitLocOpBuilder &b,
                           ConversionPatternRewriter &rewriter,
                           func::FuncOp graphBuilder, Value handle) const {
    MLIRContext *ctx = getContext();
    ModuleOp module = op->getParentOfType<ModuleOp>();

    // Call a builder function to get an operation graph.
    auto operationGraph = b.create<func::CallOp>(
        graphBuilder.getSymName(), OperationGraphType::get(ctx), handle);

    // Build an executable from the operation graph.
    auto executableCreate = api->getExecutableCreateFunction(rewriter, module);
    auto executable = b.create<func::CallOp>(executableCreate.getSymName(),
                                             ExecutableType::get(ctx),
                                             operationGraph.getResult(0));

    return executable.getResult(0);
  }

  Value getGlobalExecutable(cudnn::CallOp op, ImplicitLocOpBuilder &b,
                            ConversionPatternRewriter &rewriter,
                            func::FuncOp graphBuilder,
                            IREE::Util::GlobalLoadOp loadHandle) const {
    MLIRContext *ctx = getContext();

    // Add a global that will hold the cuDNN executable constructed from a graph
    // builder right after the builder definition.
    b.setInsertionPointAfter(graphBuilder);
    auto global = b.create<IREE::Util::GlobalOp>(
        (op.getCallee() + ".executable").str(),
        /*isMutable=*/false, ExecutableType::get(ctx));

    // Construct operation graph and executable at module initialization time.
    auto initializer = b.create<IREE::Util::InitializerOp>();
    b.setInsertionPointToStart(initializer.addEntryBlock());

    auto handle = b.create<IREE::Util::GlobalLoadOp>(HandleType::get(ctx),
                                                     loadHandle.getGlobal());
    auto executable = getLocalExecutable(op, b, rewriter, graphBuilder, handle);
    b.create<IREE::Util::GlobalStoreOp>(executable, global.getSymName());
    b.create<IREE::Util::InitializerReturnOp>();

    // Load executable from a global right before the original op.
    b.setInsertionPoint(op);
    auto loadedExecutable = b.create<IREE::Util::GlobalLoadOp>(
        ExecutableType::get(ctx), global.getSymName());

    return loadedExecutable.getResult();
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

    // Get the cuDNN handle from the parent block. We rely on the fact that
    // parent graph operation was converted earlier.
    SmallVector<Value> args = {op->getBlock()->getArgument(0)};
    args.append(adaptor.getOperands().begin(), adaptor.getOperands().end());
    assert(args[0].getType().isa<HandleType>() && "expected cuDNN handle type");

    // Create an operation graph from the returned tensor results.
    auto opGraphCreate = api->getOperationGraphCreateFunction(rewriter, module);
    auto opGraph = b.create<func::CallOp>(opGraphCreate.getSymName(),
                                          OperationGraphType::get(ctx), args);
    b.create<func::ReturnOp>(opGraph->getResults());

    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// cuDNN Pointwise Unary operations lowering
//===----------------------------------------------------------------------===//

template <typename T>
struct ConvertCudnnUnaryOp : public CudnnOpConversionPattern<T> {
  using CudnnOpConversionPattern<T>::CudnnOpConversionPattern;
  using OpAdaptor = typename CudnnOpConversionPattern<T>::OpAdaptor;

  LogicalResult matchAndRewrite(
      T op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    MLIRContext *ctx = rewriter.getContext();
    ImplicitLocOpBuilder b(op->getLoc(), rewriter);

    auto f32 = rewriter.getF32Type();

    SmallVector<Value> args = {
        adaptor.getX(),
        b.create<arith::ConstantFloatOp>(adaptor.getAlpha(), f32),
        b.create<arith::ConstantIntOp>(IsVirtual(op.getY()), 32),
    };

    auto fn = this->api->getPointwiseUnaryFunction(
        rewriter, op->template getParentOfType<ModuleOp>(),
        op->getName().stripDialect());
    rewriter.replaceOpWithNewOp<func::CallOp>(op, fn.getSymName(),
                                              CudnnTensorType::get(ctx), args);

    return success();
  }
};

//===----------------------------------------------------------------------===//
// cudnn.sqrt
//===----------------------------------------------------------------------===//

struct ConvertCudnnSqrtOp : public ConvertCudnnUnaryOp<SqrtOp> {
  using ConvertCudnnUnaryOp::ConvertCudnnUnaryOp;
};

//===----------------------------------------------------------------------===//
// cuDNN Pointwise Binary operations lowering
//===----------------------------------------------------------------------===//

template <typename T>
struct ConvertCudnnBinaryOp : public CudnnOpConversionPattern<T> {
  using CudnnOpConversionPattern<T>::CudnnOpConversionPattern;
  using OpAdaptor = typename CudnnOpConversionPattern<T>::OpAdaptor;

  LogicalResult matchAndRewrite(
      T op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    MLIRContext *ctx = rewriter.getContext();
    ImplicitLocOpBuilder b(op->getLoc(), rewriter);

    auto f32 = rewriter.getF32Type();

    SmallVector<Value> args = {
        adaptor.getX(),
        b.create<arith::ConstantFloatOp>(adaptor.getAlpha(), f32),
        adaptor.getB(),
        b.create<arith::ConstantFloatOp>(adaptor.getAlpha2(), f32),
        b.create<arith::ConstantIntOp>(IsVirtual(op.getY()), 32),
    };

    auto fn = this->api->getPointwiseBinaryFunction(
        rewriter, op->template getParentOfType<ModuleOp>(),
        op->getName().stripDialect());
    rewriter.replaceOpWithNewOp<func::CallOp>(op, fn.getSymName(),
                                              CudnnTensorType::get(ctx), args);

    return success();
  }
};

//===----------------------------------------------------------------------===//
// cudnn.add
//===----------------------------------------------------------------------===//

struct ConvertCudnnAddOp : public ConvertCudnnBinaryOp<AddOp> {
  using ConvertCudnnBinaryOp::ConvertCudnnBinaryOp;
};

//===----------------------------------------------------------------------===//
// cudnn.div
//===----------------------------------------------------------------------===//

struct ConvertCudnnDivOp : public ConvertCudnnBinaryOp<DivOp> {
  using ConvertCudnnBinaryOp::ConvertCudnnBinaryOp;
};

//===----------------------------------------------------------------------===//
// cudnn.max
//===----------------------------------------------------------------------===//

struct ConvertCudnnMaxOp : public ConvertCudnnBinaryOp<MaxOp> {
  using ConvertCudnnBinaryOp::ConvertCudnnBinaryOp;
};

//===----------------------------------------------------------------------===//
// cudnn.mul
//===----------------------------------------------------------------------===//

struct ConvertCudnnMulOp : public ConvertCudnnBinaryOp<MulOp> {
  using ConvertCudnnBinaryOp::ConvertCudnnBinaryOp;
};

//===----------------------------------------------------------------------===//
// cudnn.sub
//===----------------------------------------------------------------------===//

struct ConvertCudnnSubOp : public ConvertCudnnBinaryOp<SubOp> {
  using ConvertCudnnBinaryOp::ConvertCudnnBinaryOp;
};

//===----------------------------------------------------------------------===//
// cudnn.bias
//===----------------------------------------------------------------------===//

struct ConvertCudnnBiasOp : public CudnnOpConversionPattern<BiasOp> {
  using CudnnOpConversionPattern::CudnnOpConversionPattern;

  LogicalResult matchAndRewrite(
      BiasOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    MLIRContext *ctx = rewriter.getContext();
    ImplicitLocOpBuilder b(op->getLoc(), rewriter);

    SmallVector<Value> args = {
        adaptor.getX(),
        adaptor.getB(),
        b.create<arith::ConstantIntOp>(IsVirtual(op.getY()), 32),
    };

    auto bias = api->getBiasFunction(rewriter, op->getParentOfType<ModuleOp>());
    rewriter.replaceOpWithNewOp<func::CallOp>(op, bias.getSymName(),
                                              CudnnTensorType::get(ctx), args);

    return success();
  }
};

//===----------------------------------------------------------------------===//
// cudnn.convolution
//===----------------------------------------------------------------------===//

struct ConvertCudnnConvolutionOp
    : public CudnnOpConversionPattern<ConvolutionOp> {
  using CudnnOpConversionPattern::CudnnOpConversionPattern;

  LogicalResult matchAndRewrite(
      ConvolutionOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    MLIRContext *ctx = rewriter.getContext();
    ImplicitLocOpBuilder b(op->getLoc(), rewriter);

    int32_t mode = static_cast<int32_t>(op.getMode());

    // Prepare arguments for convolution API call.
    SmallVector<Value> args(adaptor.getOperands());
    auto pushBack = [&](llvm::ArrayRef<int64_t> values) {
      for (int64_t value : values)
        args.push_back(b.create<arith::ConstantIntOp>(value, 64));
    };
    pushBack(op.getSpatialStride());
    pushBack(op.getPrePadding());
    pushBack(op.getPostPadding());
    pushBack(op.getDilation());
    args.push_back(b.create<arith::ConstantIntOp>(IsVirtual(op.getY()), 32));
    args.push_back(b.create<arith::ConstantIntOp>(mode, 32));

    // Replace convolution operation with a convolution API call.
    auto convolution = api->getConvolutionFunction(
        rewriter, op->getParentOfType<ModuleOp>(), op.getSpatialDimCount());
    rewriter.replaceOpWithNewOp<func::CallOp>(op, convolution.getSymName(),
                                              CudnnTensorType::get(ctx), args);

    return success();
  }
};

}  // namespace

void populateCudnnToRuntimePatterns(mlir::TypeConverter &typeConverter,
                                    mlir::RewritePatternSet &patterns) {
  MLIRContext *ctx = patterns.getContext();
  auto api = std::make_shared<CudnnAPI>();

  //===--------------------------------------------------------------------===//
  // High level cuDNN library integration operations
  //===--------------------------------------------------------------------===//

  patterns.insert<ConvertCudnnHandleOp>(typeConverter, ctx, api);
  patterns.insert<ConvertCudnnGraphOp>(typeConverter, ctx, api);
  patterns.insert<ConvertCudnnReturnOp>(typeConverter, ctx, api);
  patterns.insert<ConvertCudnnCallOp>(typeConverter, ctx, api);

  //===--------------------------------------------------------------------===//
  // Pointwise unary operations
  //===--------------------------------------------------------------------===//

  patterns.insert<ConvertCudnnSqrtOp>(typeConverter, ctx, api);

  //===--------------------------------------------------------------------===//
  // Pointwise binary operations
  //===--------------------------------------------------------------------===//

  patterns.insert<ConvertCudnnAddOp, ConvertCudnnDivOp, ConvertCudnnMaxOp,
                  ConvertCudnnMulOp, ConvertCudnnSubOp>(typeConverter, ctx,
                                                        api);

  //===--------------------------------------------------------------------===//
  // The rest of cuDNN operations
  //===--------------------------------------------------------------------===//

  patterns.insert<ConvertCudnnBiasOp>(typeConverter, ctx, api);
  patterns.insert<ConvertCudnnConvolutionOp>(typeConverter, ctx, api);
}

}  // namespace openxla::compiler::nvgpu::cudnn
