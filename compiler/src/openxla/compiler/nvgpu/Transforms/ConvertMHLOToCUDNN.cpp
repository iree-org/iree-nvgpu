// Copyright 2023 The OpenXLA Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <algorithm>
#include <numeric>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "openxla/compiler/nvgpu/Dialect/CUDNN/IR/CUDNNDialect.h"
#include "openxla/compiler/nvgpu/Dialect/CUDNN/IR/CUDNNOps.h"
#include "openxla/compiler/nvgpu/Transforms/Passes.h"
#include "stablehlo/dialect/StablehloOps.h"

#define GEN_PASS_DEF_CONVERTMHLOTOCUDNN
#include "openxla/compiler/nvgpu/Transforms/Passes.h.inc"

using namespace mlir;

namespace openxla::compiler::nvgpu::cudnn {

static TensorType getCudnnTensorType(mlir::TensorType tensor_type) {
  return TensorType::get(tensor_type.getShape(), tensor_type.getElementType());
}

// Outlines all transitive ops defining 'result' into a cudnn.graph op to be
// called with `arguments`.
static LogicalResult outlineToGraph(
    TypedValue<mlir::TensorType> result,
    ArrayRef<TypedValue<mlir::TensorType>> arguments,
    PatternRewriter& rewriter) {
  Operation* root = result.getDefiningOp();
  if (!root)
    return rewriter.notifyMatchFailure(result.getLoc(), "expected def by op");
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
  rewriter.create<ReturnOp>(root->getLoc(), results);

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

// Matches any clamp 'op' which can be converted to relu and outlines it into
// a cudnn.graph op.
// TODO(chsigg): extend this to match a set of ops (determined through a cost
// model) to run as a single graph.
static LogicalResult outlineClamp(stablehlo::ClampOp op,
                                  PatternRewriter& rewriter) {
  if (failed(matchRelu(op, rewriter))) return failure();
  return outlineToGraph(op.getResult(), op.getOperand(), rewriter);
}

// Converts a clamp 'op' into a cudnn.pointwise_relu.
static LogicalResult convertClamp(stablehlo::ClampOp op,
                                  PatternRewriter& rewriter) {
  if (!op->getParentOfType<GraphOp>())
    return rewriter.notifyMatchFailure(op, "expected child of graph");
  auto min_or = matchRelu(op, rewriter);
  if (failed(min_or)) return failure();
  auto operand = op.getOperand();
  TensorType tensor_type = getCudnnTensorType(operand.getType());
  auto cast_op = rewriter.create<UnrealizedConversionCastOp>(
      operand.getLoc(), tensor_type, operand);
  rewriter.replaceOpWithNewOp<PointWiseReluOp>(
      op, tensor_type, cast_op.getResult(0), tensor_type.getElementType(),
      APFloat(min_or->convertToDouble()));
  return success();
}

namespace {

class ConvertMHLOToCUDNN : public ::impl::ConvertMHLOToCUDNNBase<
    ConvertMHLOToCUDNN> {
 public:
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    // TODO(chsigg): don't outline and convert in the same pattern application.
    // This currently works because the first expects the parent to be a child
    // of func.func, the second one a child of cudnn.graph.
    patterns.add(&outlineClamp);
    patterns.add(&convertClamp);
    if (failed(::applyPatternsAndFoldGreedily(getOperation(),
                                              std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>> createConvertMHLOToCUDNNPass() {
  return std::make_unique<ConvertMHLOToCUDNN>();
}

}  // namespace openxla::compiler::nvgpu::cudnn
