// Copyright 2023 The OpenXLA Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <iree/compiler/Dialect/HAL/IR/HALTypes.h>
#include <openxla/compiler/nvgpu/Dialect/TritonFlow/IR/TritonFlowOps.h>
#include <triton/Dialect/Triton/IR/Dialect.h>

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <optional>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Module.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Support/LogicalResult.h"
#include "openxla/compiler/nvgpu/Dialect/TritonFlow/Transforms/PassDetail.h"
#include "openxla/compiler/nvgpu/Dialect/TritonFlow/Transforms/Passes.h"

#define GEN_PASS_DEF_OUTLINETRITONCALLS
#include "openxla/compiler/nvgpu/Dialect/TritonFlow/Transforms/Passes.h.inc"

namespace IREE = mlir::iree_compiler::IREE;

using namespace mlir;

namespace openxla::compiler::nvgpu::tritonflow {

// Returns true if the type uses `pushConstants` IREE ABI.
static bool isPushConstant(Type type) { return type.isIntOrIndex(); }

// Returns the IREE HAL pipeline layout inferred from the Triton call operation.
// TODO(ezhulenev): Add support for tied results.
static IREE::HAL::PipelineLayoutAttr getPipelineLayout(tritonflow::CallOp call,
                                                       triton::FuncOp callee) {
  MLIRContext* ctx = callee->getContext();

  // Scalar arguments passed as dispatch constants.
  int64_t pushConstants =
      llvm::count_if(callee.getArgumentTypes(), isPushConstant);

  // Bindings for all tensor (pointer) arguments.
  llvm::SmallVector<IREE::HAL::DescriptorSetBindingAttr> bindings;

  for (auto arg : llvm::enumerate(call.getArguments())) {
    // If argument is not tied to any results we assume it's read only. It's
    // undefined behavior if Triton implementation writes to the argument.
    int64_t index = call.getTiedOperandsIndexAndLength().first + arg.index();
    std::optional<IREE::HAL::DescriptorFlags> flags =
        !call.isOperandTied(index)
            ? std::optional(IREE::HAL::DescriptorFlags::ReadOnly)
            : std::nullopt;

    if (auto tensor = arg.value().getType().dyn_cast<RankedTensorType>()) {
      bindings.push_back(IREE::HAL::DescriptorSetBindingAttr::get(
          ctx, /*ordinal=*/bindings.size(),
          IREE::HAL::DescriptorType::StorageBuffer, flags));
    }
  }

  for (auto ret : call.getResults()) {
    // Skip tied results as they alrady passed in as arguments.
    if (call.getTiedResultOperand(ret)) continue;

    if (auto tensor = ret.getType().dyn_cast<RankedTensorType>()) {
      bindings.push_back(IREE::HAL::DescriptorSetBindingAttr::get(
          ctx, /*ordinal=*/bindings.size(),
          IREE::HAL::DescriptorType::StorageBuffer, std::nullopt));
    }
  }

  return IREE::HAL::PipelineLayoutAttr::get(
      ctx, pushConstants,
      IREE::HAL::DescriptorSetLayoutAttr::get(ctx, /*ordinal=*/0, bindings));
}

// Updates Triton function signature to be compatible with IREE dispatch ABI. If
// function type was updated, returns an argument permutation that was applied.
static std::optional<SmallVector<int64_t>> updateTritonFunctionForAbi(
    triton::FuncOp callee) {
  MLIRContext* ctx = callee->getContext();

  // We need to compute permutation of function arguments to update callsites.
  struct IndexedArg {
    size_t index;
    Type type;
  };

  SmallVector<IndexedArg> args;
  for (auto indexed : llvm::enumerate(callee.getArgumentTypes()))
    args.push_back({indexed.index(), indexed.value()});

  // Partition arguments types, so that all scalars pushed to the end.
  auto firstScalarArg = std::stable_partition(
      args.begin(), args.end(),
      [](IndexedArg indexed) { return !isPushConstant(indexed.type); });

  // Function already has an ABI-compatibe signature.
  if (firstScalarArg == args.end() ||
      std::distance(args.begin(), firstScalarArg) == firstScalarArg->index) {
    return std::nullopt;
  }

  // Types according to the new ABI and a permutation to get there.
  SmallVector<Type> abiTypes;
  SmallVector<int64_t> abiPermutation(args.size());

  // Add new block arguments accoding to the ABI signature, and forward all
  // users of original arguments to the new args.
  Block& entryBlock = callee.front();

  for (auto arg : llvm::enumerate(args)) {
    size_t index = arg.index();
    IndexedArg value = arg.value();

    abiTypes.push_back(value.type);
    abiPermutation[value.index] = index;

    auto originalArg = entryBlock.getArgument(value.index);
    auto abiArg = entryBlock.addArgument(value.type, originalArg.getLoc());
    originalArg.replaceAllUsesWith(abiArg);
  }

  // Erase original types.
  entryBlock.eraseArguments(0, args.size());

  // Update function signature to the new type.
  auto abiFunctionType = FunctionType::get(ctx, abiTypes, TypeRange());
  callee.setFunctionType(abiFunctionType);

  // Update function argument attributes to match new basic block arguments.
  SmallVector<DictionaryAttr> argAttrs;
  SmallVector<DictionaryAttr> abiArgAttrs(args.size());
  callee.getAllArgAttrs(argAttrs);
  for (unsigned i = 0; i < args.size(); ++i) {
    abiArgAttrs[abiPermutation[i]] = argAttrs[i];
  }
  callee.setAllArgAttrs(abiArgAttrs);

  return abiPermutation;
}

// Updates Triton call operation according to the arguments permutation computed
// by the `updateTritonFunctionForAbi` function defined above.
static void updateTritonCallForAbi(tritonflow::CallOp call,
                                   SmallVector<int64_t> permutaion) {
  SmallVector<Value> args = llvm::to_vector(call.getArguments());
  SmallVector<Value> abiArgs(call.getArguments().size());

  // Number of result tensors not tied to any of the operands.
  size_t numUntiedResults = llvm::count_if(
      call.getResults(),
      [&](Value result) { return !call.getTiedResultOperand(result); });

  for (unsigned i = 0; i < args.size(); ++i) {
    int64_t newArgIdx = permutaion[i];

    // In Triton function type scalar (constant) arguments are passed after
    // destination pointers for results, so we have to adjust index for that.
    if (newArgIdx >= args.size()) newArgIdx -= numUntiedResults;

    abiArgs[newArgIdx] = args[i];
  }

  call.getArgumentsMutable().assign(abiArgs);

  // Tie results to the new operands after updating arguments.
  if (auto tiedOperands = call.getTiedOperands()) {
    SmallVector<int64_t> abiTiedOperands;

    for (auto attr : tiedOperands->getValue()) {
      auto idx = attr.cast<IntegerAttr>().getInt();
      abiTiedOperands.push_back(idx < 0 ? idx : permutaion[idx]);
    }

    call.setTiedOperandsAttr(
        OpBuilder(call).getIndexArrayAttr(abiTiedOperands));
  }
}

static LogicalResult outlineTritonFlowCallOp(tritonflow::CallOp call,
                                             SymbolTable& symTable) {
  // Get the Triton function callee.
  auto callee = symTable.lookup<triton::FuncOp>(call.getCallee());
  ImplicitLocOpBuilder b(callee->getLoc(), call->getParentOfType<ModuleOp>());

  // Update function and a call operation to be ABI compatible.
  if (auto permutation = updateTritonFunctionForAbi(callee)) {
    updateTritonCallForAbi(call, *permutation);
  }

  // Get the IREE pipeline layout (aka ABI) from the callee.
  IREE::HAL::PipelineLayoutAttr layout = getPipelineLayout(call, callee);

  // Create executable right after the original Triton function.
  b.setInsertionPointAfter(callee);
  auto executableOp =
      b.create<ExecutableOp>(callee.getSymName() + ".executable");

  // Create the inner ModuleOp that contains the original Triton function.
  b.setInsertionPointToStart(&executableOp.getBlock());
  auto innerModule = b.create<ModuleOp>();

  // Move callee into the inner module block.
  auto& innerModuleBlock = innerModule.getBodyRegion().front();
  callee->moveBefore(&innerModuleBlock, innerModuleBlock.end());

  // Export Triton function from the executable.
  b.setInsertionPointToStart(&executableOp.getBlock());
  auto exportExecutableOp = b.create<ExecutableExportOp>(
      callee.getSymName(), call.getCalleeAttr(), layout);

  // Replace call operation with a Triton executable dispatch.
  b.setLoc(call.getLoc());
  b.setInsertionPoint(call);

  auto dispatchOp = b.create<DispatchOp>(
      exportExecutableOp, call.getGrid(), call.getResultTypes(),
      call.getResultDims(), call.getArguments(), call.getArgumentDims(),
      call.getTiedOperands().value_or(nullptr));

  call->replaceAllUsesWith(dispatchOp->getResults());
  call->erase();

  return success();
}

class OutlineTritonCalls
    : public ::impl::OutlineTritonCallsBase<OutlineTritonCalls> {
 public:
  void runOnOperation() override {
    SmallVector<tritonflow::CallOp> calls;
    getOperation()->walk([&](tritonflow::CallOp op) { calls.push_back(op); });
    if (calls.empty()) return;

    SymbolTable symTable(getOperation());
    for (auto call : calls) {
      if (failed(outlineTritonFlowCallOp(call, symTable)))
        return signalPassFailure();
    }
  }
};

std::unique_ptr<OperationPass<ModuleOp>> createOutlineTritonCallsPass() {
  return std::make_unique<OutlineTritonCalls>();
}

}  // namespace openxla::compiler::nvgpu::tritonflow
