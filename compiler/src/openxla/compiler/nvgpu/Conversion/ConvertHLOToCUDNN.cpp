// Copyright 2023 The OpenXLA Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "openxla/compiler/nvgpu/Conversion/ConvertHLOToCUDNN.h"

#include <algorithm>
#include <iterator>

#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Support/LogicalResult.h"
#include "openxla/compiler/nvgpu/Dialect/CUDNN/IR/CUDNNDialect.h"
#include "openxla/compiler/nvgpu/Dialect/CUDNN/IR/CUDNNOps.h"
#include "openxla/compiler/nvgpu/Dialect/CUDNN/IR/CUDNNTypes.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace openxla::compiler::nvgpu::cudnn {

using namespace mlir;
using namespace mlir::iree_compiler;

using TensorValue = TypedValue<TensorType>;

static CudnnTensorType getCudnnTensorType(mlir::TensorType type,
                                          Layout layout) {
  auto shape = [&]() -> SmallVector<int64_t> {
    auto shape = type.getShape();
    switch (layout) {
      case Layout::NCHW:
      case Layout::KCHW:
        return {shape.begin(), shape.end()};
      case Layout::NHWC:
      case Layout::KHWC:
        return {shape[0], shape[3], shape[1], shape[2]};
    }
    return {};
  }();
  return CudnnTensorType::get(shape, type.getElementType(), layout);
}

static FailureOr<Layout> getCudnnTensorLayout(int64_t batchDim,
                                              int64_t featureDim) {
  if (batchDim != 0) return failure();
  if (featureDim == 1) return Layout::NCHW;
  if (featureDim == 3) return Layout::NHWC;
  return failure();
}

static FailureOr<Layout> getCudnnKernelLayout(int64_t inputDim,
                                              int64_t outputDim,
                                              MLIRContext* ctx) {
  if (outputDim != 0) return failure();
  // Return input/output layout instead of the kernel layout so that casts
  // cancel themselves. The actual layouts are the same and the lowering and
  // runtime implementation handle it correctly. TODO(chsigg): revert when
  // layout propagation is implemented.
  if (inputDim == 1) return Layout::NCHW;  // KCHW
  if (inputDim == 3) return Layout::NHWC;  // KHWC
  return failure();
}

// Outlines all transitive ops defining 'result' into a cudnn.graph op and calls
// it with `arguments`.
static LogicalResult outlineToGraph(TensorValue result,
                                    ArrayRef<TensorValue> arguments,
                                    FunctionType funcType,
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
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(func);
  GraphOp graph = rewriter.create<GraphOp>(root->getLoc(), funcType,
                                           /*arg_attrs=*/ArrayAttr{},
                                           /*res_attrs=*/ArrayAttr{});
  SymbolTable symtab(func->getParentOp());
  graph.setName(root->getName().getStringRef());
  symtab.insert(graph);

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

struct ConvLayouts {
  Layout input, kernel, output;
};

// Returns conv layouts, if it can be converted to cudnn.convolution.
static FailureOr<ConvLayouts> matchConv(stablehlo::ConvolutionOp op,
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

  FailureOr<Layout> inputLayout = getCudnnTensorLayout(
      dims.getInputBatchDimension(), dims.getInputFeatureDimension());
  if (failed(inputLayout))
    return rewriter.notifyMatchFailure(op, "expected input to be NCHW or NHWC");

  FailureOr<Layout> kernelLayout = getCudnnKernelLayout(
      dims.getKernelInputFeatureDimension(),
      dims.getKernelOutputFeatureDimension(), op->getContext());
  if (failed(kernelLayout)) {
    return rewriter.notifyMatchFailure(
        op, "expected kernel to be KCHW, KHWC or HWCK");
  }

  FailureOr<Layout> outputLayout = getCudnnTensorLayout(
      dims.getOutputBatchDimension(), dims.getOutputFeatureDimension());
  if (failed(outputLayout)) {
    return rewriter.notifyMatchFailure(op,
                                       "expected output to be NCHW or NHWC");
  }

  return {{*inputLayout, *kernelLayout, *outputLayout}};
}

// Matches defining ops starting from 'result' and returns the graph inputs.
static LogicalResult outlineToGraph(TensorValue result,
                                    PatternRewriter& rewriter) {
  // The layout of the first convolution output, if we encounter such an op.
  // Eventually, this should be replaced by layout propagation.
  std::optional<Layout> layout;
  SmallVector<TensorValue> arguments = {result};
  for (size_t i = 0; i < arguments.size();) {
    Operation* defOp = arguments[i].getDefiningOp();
    if (defOp == nullptr) {
      ++i;
      continue;
    }
    if (auto op = dyn_cast<stablehlo::ClampOp>(defOp)) {
      if (succeeded(matchRelu(op, rewriter))) {
        arguments[i] = op.getOperand();
      } else {
        ++i;
      }
      continue;
    }
    if (auto op = dyn_cast<stablehlo::ConvolutionOp>(defOp)) {
      auto layouts = matchConv(op, rewriter);
      if (succeeded(layouts) && (!layout || layout == layouts->output)) {
        layout = layouts->output;
        arguments[i] = op.getLhs();
        arguments.push_back(op.getRhs());
      } else {
        ++i;
      }
      continue;
    }
    ++i;
  }
  if (arguments.size() == 1 && arguments.front() == result) return failure();
  auto getType = [&](TensorValue operand) -> Type {
    return getCudnnTensorType(operand.getType(), layout.value_or(Layout::NCHW));
  };
  FunctionType funcType = rewriter.getFunctionType(
      to_vector(map_range(arguments, getType)), getType(result));
  return outlineToGraph(result, arguments, funcType, rewriter);
}

// Pattern entry point to outline a cudnn.graph with 'root'.
template <typename OpTy>
static LogicalResult outlineToGraph(OpTy root, PatternRewriter& rewriter) {
  return outlineToGraph(root.getResult(), rewriter);
}

static Value castToCudnnTensor(TensorValue value, Layout layout,
                               PatternRewriter& rewriter) {
  CudnnTensorType tensorType = getCudnnTensorType(value.getType(), layout);
  auto castOp = rewriter.create<UnrealizedConversionCastOp>(value.getLoc(),
                                                            tensorType, value);
  return castOp.getResult(0);
}

// Converts a clamp 'op' into a cudnn.pointwise_relu.
static LogicalResult convertClamp(stablehlo::ClampOp op,
                                  PatternRewriter& rewriter) {
  auto graph = op->getParentOfType<GraphOp>();
  if (!graph) return rewriter.notifyMatchFailure(op, "expected child of graph");
  auto minOr = matchRelu(op, rewriter);
  if (failed(minOr)) return failure();
  // Lookup layout from graph result type. This assumes that the layout is
  // consistent across the graph. TODO(csigg): propagate layout to attribute.
  Layout layout =
      *cast<CudnnTensorType>(graph.getFunctionType().getResult(0)).getLayout();
  Type resultType = getCudnnTensorType(op.getType(), layout);
  Value operand = castToCudnnTensor(op.getOperand(), layout, rewriter);
  Type elementType = op.getType().getElementType();

  ReluOp reluOp =
      rewriter.create<ReluOp>(op.getLoc(), resultType, operand, elementType,
                              APFloat(minOr->convertToDouble()));
  rewriter.replaceOpWithNewOp<UnrealizedConversionCastOp>(op, resultType,
                                                          reluOp.getResult());
  return success();
}

// Converts convolution 'op' into a cudnn.convolution.
static LogicalResult convertConv(stablehlo::ConvolutionOp op,
                                 PatternRewriter& rewriter) {
  if (!op->getParentOfType<GraphOp>())
    return rewriter.notifyMatchFailure(op, "expected child of graph");
  FailureOr<ConvLayouts> layouts = matchConv(op, rewriter);
  if (failed(layouts)) return failure();

  Type resultType = getCudnnTensorType(op.getType(), layouts->output);
  Value lhs = castToCudnnTensor(op.getLhs(), layouts->input, rewriter);
  Value rhs = castToCudnnTensor(op.getRhs(), layouts->kernel, rewriter);

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

  ConvolutionOp convOp = rewriter.create<ConvolutionOp>(
      op.getLoc(), resultType, lhs, rhs, alpha, beta, spatialDimCount,
      spatialStride, prePadding, postPadding, dilation);
  rewriter.replaceOpWithNewOp<UnrealizedConversionCastOp>(op, resultType,
                                                          convOp.getResult());
  return success();
}

void populateOutlineHLOToCUDNNPatterns(mlir::RewritePatternSet& patterns) {
  patterns.add(&outlineToGraph<stablehlo::ClampOp>);
  patterns.add(&outlineToGraph<stablehlo::ConvolutionOp>);
}

void populateConvertHLOToCUDNNPatterns(mlir::RewritePatternSet& patterns) {
  patterns.add(&convertClamp);
  patterns.add(&convertConv);
}

}  // namespace openxla::compiler::nvgpu::cudnn
