// Copyright 2023 The OpenXLA Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "openxla/compiler/nvgpu/Conversion/ConvertHLOToCUDNN.h"

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

using namespace mlir;

namespace openxla::compiler::nvgpu::cudnn {

static TensorType getCudnnTensorType(mlir::TensorType tensor_type) {
  return TensorType::get(tensor_type.getShape(), tensor_type.getElementType());
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
static LogicalResult outlineToGraph(
    TypedValue<mlir::TensorType> result,
    ArrayRef<TypedValue<mlir::TensorType>> arguments,
    PatternRewriter& rewriter) {
  Operation* root = result.getDefiningOp();
  if (!root)
    return rewriter.notifyMatchFailure(result.getLoc(), "expected def by op");
  if (root->getNumResults() != 1)
    return rewriter.notifyMatchFailure(result.getLoc(), "expected one result");
  func::FuncOp func = root->getParentOfType<func::FuncOp>();
  if (!func) return rewriter.notifyMatchFailure(root, "expected child of func");

  // Collect the set of ops to clone into the region.
  SetVector<Operation*> ops;
  ops.insert(root);
  DenseSet<Value> arg_set(arguments.begin(), arguments.end());
  for (size_t index = 0; index < ops.size(); ++index) {
    Operation* op = ops[index];
    for (Value operand : op->getOperands()) {
      if (arg_set.count(operand)) continue;
      Operation* def = operand.getDefiningOp();
      if (!def)
        return rewriter.notifyMatchFailure(op, "expected operands def by op");
      ops.insert(def);
    }
  }

  // Create the cudnn.graph op with an empty region.
  auto arg_types = llvm::to_vector(
      map_range(arguments, [](TypedValue<mlir::TensorType> arg) -> Type {
        return getCudnnTensorType(arg.getType());
      }));
  FunctionType func_type =
      rewriter.getFunctionType(arg_types, getCudnnTensorType(result.getType()));
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(func);
  GraphOp graph = rewriter.create<GraphOp>(root->getLoc(), func_type,
                                           /*arg_attrs=*/ArrayAttr{},
                                           /*res_attrs=*/ArrayAttr{});
  graph.setName(root->getName().getStringRef());
  SymbolTable symtab(func->getParentOp());

  // Clone ops into region.
  rewriter.setInsertionPointToEnd(graph.addEntryBlock());
  IRMapping mapping;
  for (size_t index = 0; index < arguments.size(); ++index) {
    Value arg = arguments[index];
    auto cast_op = rewriter.create<UnrealizedConversionCastOp>(
        arg.getLoc(), arg.getType(), graph.getArgument(index));
    mapping.map(arg, cast_op.getResult(0));
  }
  ValueRange results;
  for (Operation* op : llvm::reverse(ops)) {
    results = rewriter.clone(*op, mapping)->getResults();
    mapping.map(op->getResults(), results);
  }
  auto cast_op = rewriter.create<UnrealizedConversionCastOp>(
      result.getLoc(), func_type.getResults(), results);
  rewriter.create<ReturnOp>(root->getLoc(), cast_op.getResults());

  // Replace root with cudnn.call op.
  rewriter.setInsertionPoint(root);
  rewriter.replaceOpWithNewOp<CallOp>(
      root, result.getType(), graph.getName(),
      ArrayRef<Value>(arguments.data(), arguments.size()));

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

  auto is_none_or_value = [](std::optional<DenseIntElementsAttr> attr,
                             int64_t value) {
    return !attr || llvm::all_of(attr->getValues<APInt>(), [&](APInt value) {
      return value.getSExtValue() == value;
    });
  };
  if (!is_none_or_value(op.getLhsDilation(), 1))
    return rewriter.notifyMatchFailure(op, "expected lhs_dilation to be 1");
  if (!is_none_or_value(op.getRhsDilation(), 1))
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
                           dims.getInputFeatureDimension()) != Layout::NCHW)
    return rewriter.notifyMatchFailure(op, "expected input to be NCHW");
  if (getCudnnTensorLayout(dims.getOutputBatchDimension(),
                           dims.getOutputFeatureDimension()) != Layout::NCHW)
    return rewriter.notifyMatchFailure(op, "expected output to be NCHW");
  // TODO(chsigg): support HWIO.
  if (getCudnnTensorLayout(dims.getKernelOutputFeatureDimension(),
                           dims.getKernelInputFeatureDimension()) !=
      Layout::NCHW)
    return rewriter.notifyMatchFailure(op, "expected kernel to be OIHW");

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

static Value castToCudnnTensor(TypedValue<mlir::TensorType> value,
                               PatternRewriter& rewriter) {
  TensorType tensor_type = getCudnnTensorType(value.getType());
  auto cast_op = rewriter.create<UnrealizedConversionCastOp>(
      value.getLoc(), tensor_type, value);
  return cast_op.getResult(0);
}

// Converts a clamp 'op' into a cudnn.pointwise_relu.
static LogicalResult convertClamp(stablehlo::ClampOp op,
                                  PatternRewriter& rewriter) {
  if (!op->getParentOfType<GraphOp>())
    return rewriter.notifyMatchFailure(op, "expected child of graph");
  auto min_or = matchRelu(op, rewriter);
  if (failed(min_or)) return failure();
  Type result_type = getCudnnTensorType(op.getType());
  Value operand = castToCudnnTensor(op.getOperand(), rewriter);
  Type element_type = op.getType().getElementType();
  rewriter.replaceOpWithNewOp<ReluOp>(op, result_type, operand, element_type,
                                      APFloat(min_or->convertToDouble()));
  return success();
}
// Converts convolution 'op' into a cudnn.convolution.
static LogicalResult convertConv(stablehlo::ConvolutionOp op,
                                 PatternRewriter& rewriter) {
  if (!op->getParentOfType<GraphOp>())
    return rewriter.notifyMatchFailure(op, "expected child of graph");
  if (failed(matchConv(op, rewriter))) return failure();
  Type result_type = getCudnnTensorType(op.getType());
  Value lhs = castToCudnnTensor(op.getLhs(), rewriter);
  Value rhs = castToCudnnTensor(op.getRhs(), rewriter);

  APFloat alpha(1.0f);
  APFloat beta(0.0f);

  uint32_t spatial_dim_count =
      op.getDimensionNumbers().getInputSpatialDimensions().size();
  auto get_attr_or = [&](std::optional<DenseIntElementsAttr> attr,
                         int64_t value) {
    if (!attr) return SmallVector<int64_t>(spatial_dim_count, value);
    SmallVector<int64_t> values;
    llvm::transform(attr->getValues<APInt>(), std::back_inserter(values),
                    [](APInt stride) { return stride.getSExtValue(); });
    return values;
  };
  SmallVector<int64_t> spatial_stride = get_attr_or(op.getWindowStrides(), 1);
  SmallVector<int64_t> pre_padding = get_attr_or(op.getPadding(), 0);
  SmallVector<int64_t> post_padding(spatial_dim_count, 0);
  SmallVector<int64_t> dilation = get_attr_or(op.getRhsDilation(), 1);

  rewriter.replaceOpWithNewOp<ConvolutionOp>(
      op, result_type, lhs, rhs, alpha, beta, spatial_dim_count, spatial_stride,
      pre_padding, post_padding, dilation);
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
