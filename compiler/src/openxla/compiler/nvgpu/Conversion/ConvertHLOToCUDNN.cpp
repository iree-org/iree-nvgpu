// Copyright 2023 The OpenXLA Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "openxla/compiler/nvgpu/Conversion/ConvertHLOToCUDNN.h"

#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "openxla/compiler/nvgpu/Dialect/CUDNN/IR/CUDNNDialect.h"
#include "openxla/compiler/nvgpu/Dialect/CUDNN/IR/CUDNNOps.h"
#include "openxla/compiler/nvgpu/Dialect/CUDNN/IR/CUDNNTypes.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace openxla::compiler::nvgpu::cudnn {

using namespace mlir;
using namespace mlir::iree_compiler;

static CudnnTensorType getCudnnTensorType(TensorType tensor_type) {
  // Reshuffle shape accoring to NHWC layout.
  // TODO: Generalize this to other layouts.
  ArrayRef<int64_t> shape = tensor_type.getShape();
  assert(shape.size() == 4 && "expect 4 dims for NHWC layout");
  SmallVector<int64_t> new_shape = {shape[0], shape[3], shape[1], shape[2]};

  return CudnnTensorType::get(new_shape, tensor_type.getElementType(),
                              Layout::NHWC);
}

static FailureOr<Layout> getCudnnTensorLayout(int64_t batch_dim,
                                              int64_t feature_dim) {
  if (batch_dim != 0) return failure();
  if (feature_dim == 1) return Layout::NCHW;
  if (feature_dim == 3) return Layout::NHWC;
  return failure();
}

// Outlines all transitive ops defining 'result' into a cudnn.graph op and calls
// it with `arguments`.
static LogicalResult outlineToGraph(TypedValue<TensorType> result,
                                    ArrayRef<TypedValue<TensorType>> arguments,
                                    PatternRewriter& rewriter) {
  Operation* root = result.getDefiningOp();
  if (!root)
    return rewriter.notifyMatchFailure(result.getLoc(), "expected def by op");
  if (root->getNumResults() != 1)
    return rewriter.notifyMatchFailure(result.getLoc(), "expected one result");
  auto func = root->getParentOfType<func::FuncOp>();
  if (!func) return rewriter.notifyMatchFailure(root, "expected child of func");

  // Collect the set of ops to clone into the region.
  SetVector<Operation*> ops;
  ops.insert(root);
  DenseSet<Value> argSet(arguments.begin(), arguments.end());
  for (size_t index = 0; index < ops.size(); ++index) {
    Operation* op = ops[index];
    for (Value operand : op->getOperands()) {
      if (argSet.count(operand)) continue;
      Operation* def = operand.getDefiningOp();
      if (!def)
        return rewriter.notifyMatchFailure(op, "expected operands def by op");
      ops.insert(def);
    }
  }

  // Create the cudnn.graph op with an empty region.
  auto argTypes = llvm::to_vector(
      map_range(arguments, [](TypedValue<TensorType> arg) -> Type {
        return getCudnnTensorType(arg.getType());
      }));
  FunctionType funcType =
      rewriter.getFunctionType(argTypes, getCudnnTensorType(result.getType()));
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(func);
  auto graph = rewriter.create<GraphOp>(root->getLoc(), funcType,
                                        /*arg_attrs=*/ArrayAttr{},
                                        /*res_attrs=*/ArrayAttr{});
  graph.setName(root->getName().getStringRef());
  SymbolTable symtab(func->getParentOp());

  // Clone ops into region.
  rewriter.setInsertionPointToEnd(graph.addEntryBlock());
  IRMapping mapping;
  for (size_t index = 0; index < arguments.size(); ++index) {
    Value arg = arguments[index];
    auto castOp = rewriter.create<UnrealizedConversionCastOp>(
        arg.getLoc(), arg.getType(), graph.getArgument(index));
    mapping.map(arg, castOp.getResult(0));
  }
  ValueRange results;
  for (Operation* op : llvm::reverse(ops)) {
    results = rewriter.clone(*op, mapping)->getResults();
    mapping.map(op->getResults(), results);
  }
  auto castOp = rewriter.create<UnrealizedConversionCastOp>(
      result.getLoc(), funcType.getResults(), results);
  rewriter.create<ReturnOp>(root->getLoc(), castOp.getResults());

  // Replace root with cudnn.call op.
  rewriter.setInsertionPoint(root);
  auto device = rewriter.create<IREE::HAL::ExSharedDeviceOp>(root->getLoc());
  auto handle = rewriter.create<HandleOp>(root->getLoc(), device.getResult());
  rewriter.replaceOpWithNewOp<CallOp>(
      root, result.getType(), graph.getName(), handle,
      ArrayRef<Value>(arguments.begin(), arguments.size()));

  return success();
}

// Returns the 'min' value of clamp 'op', if it can be converted to relu.
static FailureOr<llvm::APFloat> matchRelu(stablehlo::ClampOp op,
                                          PatternRewriter& rewriter) {
  llvm::APFloat min = llvm::APFloat::IEEEdouble();
  if (!matchPattern(op.getMin().getDefiningOp(), m_ConstantFloat(&min))) {
    return rewriter.notifyMatchFailure(op, "expected constant min");
  }
  llvm::APFloat max = llvm::APFloat::IEEEdouble();
  if (!matchPattern(op.getMax().getDefiningOp(), m_ConstantFloat(&max)) ||
      !max.isNaN()) {
    return rewriter.notifyMatchFailure(op, "expected NaN max");
  }
  return min;
}

// Returns whether 'op' can be converted to cudnn.convolution.
LogicalResult matchConv(stablehlo::ConvolutionOp op,
                        PatternRewriter& rewriter) {
  if (op.getBatchGroupCount() != 1)
    return rewriter.notifyMatchFailure(op,
                                       "expected batch_group_count to be 1");
  if (op.getFeatureGroupCount() != 1)
    return rewriter.notifyMatchFailure(op,
                                       "expected feature_group_count to be 1");
  if (op.getPrecisionConfig())
    return rewriter.notifyMatchFailure(op, "expected no precision config");

  auto isNoneOrValue = [](std::optional<DenseIntElementsAttr> attr,
                          int64_t value) {
    return !attr || llvm::all_of(attr->getValues<APInt>(), [&](APInt value) {
      return value.getSExtValue() == value;
    });
  };
  if (!isNoneOrValue(op.getLhsDilation(), 1))
    return rewriter.notifyMatchFailure(op, "expected lhs_dilation to be 1");
  if (!isNoneOrValue(op.getRhsDilation(), 1))
    return rewriter.notifyMatchFailure(op, "expected rhs_dilation to be 1");
  if (op.getWindowReversal() &&
      llvm::any_of(op.getWindowReversal()->getValues<bool>(),
                   [](bool reversal) { return reversal; }))
    return rewriter.notifyMatchFailure(op, "expected no window_reversal");

  auto dims = op.getDimensionNumbers();
  if (dims.getInputSpatialDimensions().size() != 2 ||
      dims.getOutputSpatialDimensions().size() != 2 ||
      dims.getKernelSpatialDimensions().size() != 2) {
    return rewriter.notifyMatchFailure(op, "expected 2D convolution");
  }

  // TODO(chsigg): support NHWC layout.
  if (getCudnnTensorLayout(dims.getInputBatchDimension(),
                           dims.getInputFeatureDimension()) != Layout::NHWC)
    return rewriter.notifyMatchFailure(op, "expected input to be NCHW");
  if (getCudnnTensorLayout(dims.getOutputBatchDimension(),
                           dims.getOutputFeatureDimension()) != Layout::NHWC)
    return rewriter.notifyMatchFailure(op, "expected output to be NCHW");
  // TODO(chsigg): support HWIO.
  if (getCudnnTensorLayout(dims.getKernelOutputFeatureDimension(),
                           dims.getKernelInputFeatureDimension()) !=
      Layout::NHWC) {
    return rewriter.notifyMatchFailure(op, "expected kernel to be OIHW");
  }

  return success();
}

// Matches any clamp 'op' which can be converted to relu and outlines it into
// a cudnn.graph op.
// TODO(chsigg): extend this to match a set of ops (determined through a cost
// model) to run as a single graph.
static LogicalResult outlineClamp(stablehlo::ClampOp op,
                                  PatternRewriter& rewriter) {
  if (failed(matchRelu(op, rewriter))) return failure();
  return outlineToGraph(op.getResult(), op.getOperand(), rewriter);
}
static LogicalResult outlineConv(stablehlo::ConvolutionOp op,
                                 PatternRewriter& rewriter) {
  if (failed(matchConv(op, rewriter))) return failure();
  return outlineToGraph(op.getResult(), {op.getLhs(), op.getRhs()}, rewriter);
}

static Value castToCudnnTensor(TypedValue<TensorType> value,
                               PatternRewriter& rewriter) {
  CudnnTensorType tensorType = getCudnnTensorType(value.getType());
  auto castOp = rewriter.create<UnrealizedConversionCastOp>(value.getLoc(),
                                                            tensorType, value);
  return castOp.getResult(0);
}

// Converts a clamp 'op' into a cudnn.pointwise_relu.
static LogicalResult convertClamp(stablehlo::ClampOp op,
                                  PatternRewriter& rewriter) {
  if (!op->getParentOfType<GraphOp>())
    return rewriter.notifyMatchFailure(op, "expected child of graph");
  auto minOr = matchRelu(op, rewriter);
  if (failed(minOr)) return failure();
  Type resultType = getCudnnTensorType(op.getType());
  Value operand = castToCudnnTensor(op.getOperand(), rewriter);
  Type elementType = op.getType().getElementType();
  rewriter.replaceOpWithNewOp<ReluOp>(op, resultType, operand, elementType,
                                      APFloat(minOr->convertToDouble()));
  return success();
}

// Converts convolution 'op' into a cudnn.convolution.
static LogicalResult convertConv(stablehlo::ConvolutionOp op,
                                 PatternRewriter& rewriter) {
  if (!op->getParentOfType<GraphOp>())
    return rewriter.notifyMatchFailure(op, "expected child of graph");
  if (failed(matchConv(op, rewriter))) return failure();
  Type resultType = getCudnnTensorType(op.getType());
  Value lhs = castToCudnnTensor(op.getLhs(), rewriter);
  Value rhs = castToCudnnTensor(op.getRhs(), rewriter);

  APFloat alpha(1.0f);
  APFloat beta(0.0f);

  uint32_t spatialDimCount =
      op.getDimensionNumbers().getInputSpatialDimensions().size();
  auto getAttrOr = [](std::optional<DenseIntElementsAttr> attr, int64_t size,
                      int64_t value) -> SmallVector<int64_t> {
    if (!attr) return SmallVector<int64_t>(size, value);
    return llvm::to_vector(llvm::map_range(
        attr->getValues<APInt>(), [](APInt it) { return it.getSExtValue(); }));
  };
  SmallVector<int64_t> spatialStride =
      getAttrOr(op.getWindowStrides(), spatialDimCount, 1);
  SmallVector<int64_t> dilation =
      getAttrOr(*op.getRhsDilation(), spatialDimCount, 1);

  SmallVector<int64_t> padding =
      getAttrOr(op.getPadding(), 2 * spatialDimCount, 0);
  assert(padding.size() == 2 * spatialDimCount);
  SmallVector<int64_t> prePadding, postPadding;
  prePadding.reserve(spatialDimCount);
  postPadding.reserve(spatialDimCount);
  for (int64_t i = 0; i < spatialDimCount; ++i) {
    prePadding.push_back(padding[2 * i]);
    postPadding.push_back(padding[2 * i + 1]);
  }

  rewriter.replaceOpWithNewOp<ConvolutionOp>(
      op, resultType, lhs, rhs, alpha, beta, spatialDimCount, spatialStride,
      prePadding, postPadding, dilation);
  return success();
}

void populateOutlineHLOToCUDNNPatterns(mlir::RewritePatternSet& patterns) {
  patterns.add(&outlineClamp);
  patterns.add(&outlineConv);
}

void populateConvertHLOToCUDNNPatterns(mlir::RewritePatternSet& patterns) {
  patterns.add(&convertClamp);
  patterns.add(&convertConv);
}

}  // namespace openxla::compiler::nvgpu::cudnn
