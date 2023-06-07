// Copyright 2023 The OpenXLA Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "openxla/compiler/nvgpu/Dialect/TritonFlow/Conversion/ConvertHloToTriton.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/MemoryBufferRef.h"
#include "mhlo/IR/hlo_ops.h"
#include "mlir/Bytecode/BytecodeReader.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"
#include "openxla/compiler/nvgpu/Dialect/TritonFlow/IR/TritonFlowOps.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

namespace openxla::compiler::nvgpu::tritonflow {
namespace {

using namespace mlir;

struct ConvertCustomCallToTritonCall
    : public OpConversionPattern<mhlo::CustomCallOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      mhlo::CustomCallOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    MLIRContext *ctx = getContext();

    // Try to parse backend config into a Triton module.
    auto backendConfig = op.getBackendConfigAttr().dyn_cast<StringAttr>();
    if (!backendConfig)
      return rewriter.notifyMatchFailure(
          op, "backend config must be a string attribute");

    std::string bytecode = llvm::fromHex(backendConfig.strref());
    llvm::MemoryBufferRef buffer(bytecode, "triton.bc");

    Block block;
    ParserConfig config(ctx);
    if (failed(readBytecodeFile(buffer, &block, config)))
      return rewriter.notifyMatchFailure(
          op, "failed to read backend config as MLIR bytecode");

    // MLIR bytecode should have a single module operation in it.
    auto modules = llvm::to_vector(block.getOps<ModuleOp>());
    if (modules.size() != 1)
      return rewriter.notifyMatchFailure(
          op, "backend config must have a single module operation");

    // Module must have a single Triton function.
    auto funcs = llvm::to_vector(modules.front().getOps<triton::FuncOp>());
    if (funcs.size() != 1)
      return rewriter.notifyMatchFailure(
          op, "backend config module must have a single Triton function");

    // Dispatch grid size.
    auto grid = op->getAttrOfType<DenseI64ArrayAttr>("grid");
    if (!grid) return rewriter.notifyMatchFailure(op, "missing grid attribute");

    // Triton kernel `num_warps` required for computing the block size.
    auto numWarps = op->getAttrOfType<IntegerAttr>("num_warps");
    if (!numWarps)
      return rewriter.notifyMatchFailure(op, "missing num_warps attribute");

    // Get extra scalar arguments that have to be passed to Triton kernel.
    auto indices = op->getAttrOfType<DenseI64ArrayAttr>("scalar_args_indices");
    if (!indices)
      return rewriter.notifyMatchFailure(
          op, "missing scalar_args_indices attribuge");

    auto values = op->getAttrOfType<ArrayAttr>("scalar_args_values");
    if (!values)
      return rewriter.notifyMatchFailure(
          op, "missing scalar_args_values attribuge");

    if (indices.size() != values.size())
      return rewriter.notifyMatchFailure(
          op, "scalar arguments indices must have the same size as values");

    ImplicitLocOpBuilder b(op->getLoc(), rewriter);

    // Prepare storage for tritonflow::CallOp arguments.
    SmallVector<Value> args(op->getNumOperands() + indices.size());

    // A mapping from a custom call operand index to the `args` index after
    // adding all scalar arguments. We need to keep track of this mapping to be
    // able to convert custom call operands aliases to tied operands.
    SmallVector<int64_t> operandToArgIdx(op->getNumOperands(), -1);

    // Materialize all scalar arguments as constants.
    for (auto tuple : llvm::zip(indices.asArrayRef(), values)) {
      auto [idx, value] = tuple;
      args[idx] = b.create<arith::ConstantOp>(value.cast<TypedAttr>());
    }

    // Fill the gaps in arguments array with custom call operands (tensors).
    auto operands = llvm::to_vector(op->getOperands());
    for (int64_t i = args.size() - 1; i >= 0; --i) {
      if (!args[i]) {
        args[i] = operands.pop_back_val();
        operandToArgIdx[operands.size()] = i;
      }
    }

    // Materialize dispatch grid as constants.
    SmallVector<Value> dispatchGrid;
    for (auto size : grid.asArrayRef()) {
      dispatchGrid.push_back(
          b.create<arith::ConstantOp>(b.getIndexType(), b.getIndexAttr(size)));
    }

    // Move Triton function to the parent module.
    rewriter.inlineBlockBefore(&modules.front().getBodyRegion().front(),
                               op->getParentOp());

    // Get tied operands based on operand aliases.
    SmallVector<int64_t> tiedOperands;
    if (auto aliases = op.getOutputOperandAliases(); !aliases.empty()) {
      tiedOperands.resize(op.getNumResults(), -1);

      for (auto alias : aliases.getAsRange<mhlo::OutputOperandAliasAttr>()) {
        // TODO(ezhulenev): Add support for multiple results? How exactly alias
        // attribute will carry output index in this case? Do we even support
        // output tuples in HLO dialect?
        if (!alias.getOperandTupleIndices().empty() ||
            !alias.getOutputTupleIndices().empty())
          return rewriter.notifyMatchFailure(op, "tuples are not supported");

        tiedOperands[0] = operandToArgIdx[alias.getOperandIndex()];
      }
    }

    auto callee = FlatSymbolRefAttr::get(ctx, funcs.front().getSymName());

    rewriter.replaceOpWithNewOp<tritonflow::CallOp>(
        op, op.getResultTypes(), dispatchGrid, callee, args,
        /*argumentDims=*/ValueRange(),
        /*resultDims=*/ValueRange(),
        /*tiedOperands=*/rewriter.getIndexArrayAttr(tiedOperands));

    return success();
  }
};

}  // namespace

void populateHloToTritonPatterns(mlir::TypeConverter &typeConverter,
                                 mlir::RewritePatternSet &patterns) {
  MLIRContext *ctx = patterns.getContext();
  patterns.insert<ConvertCustomCallToTritonCall>(typeConverter, ctx);
}

}  // namespace openxla::compiler::nvgpu::tritonflow
