// Copyright 2023 The OpenXLA Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <optional>

#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Module.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Support/LogicalResult.h"
#include "openxla/compiler/nvgpu/Dialect/TritonFlow/IR/TritonFlowOps.h"
#include "openxla/compiler/nvgpu/Dialect/TritonFlow/Transforms/PassDetail.h"
#include "openxla/compiler/nvgpu/Dialect/TritonFlow/Transforms/Passes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#define GEN_PASS_DEF_REFINETRITONABI
#include "openxla/compiler/nvgpu/Dialect/TritonFlow/Transforms/Passes.h.inc"

namespace openxla::compiler::nvgpu::tritonflow {

using namespace mlir;
using namespace mlir::iree_compiler;

// Returns true if the type uses `pushConstants` IREE ABI.
static bool isPushConstant(Type type) { return type.isIntOrIndex(); }

// Returns the IREE HAL pipeline layout inferred from the Triton dispatch.
static IREE::HAL::PipelineLayoutAttr getPipelineLayout(DispatchOp dispatch,
                                                       triton::FuncOp callee) {
  MLIRContext* ctx = dispatch->getContext();

  // Scalar arguments passed as dispatch constants.
  int64_t pushConstants =
      llvm::count_if(callee.getArgumentTypes(), isPushConstant);

  // Bindings for all tensor (pointer) arguments.
  SmallVector<IREE::HAL::DescriptorSetBindingAttr> bindings;

  for (auto arg : llvm::enumerate(dispatch.getArguments())) {
    // If argument is not tied to any results we assume it's read only. It's
    // undefined behavior if Triton implementation writes to the argument.
    auto index = dispatch.getTiedOperandsIndexAndLength().first + arg.index();
    std::optional<IREE::HAL::DescriptorFlags> flags =
        !dispatch.isOperandTied(index)
            ? std::optional(IREE::HAL::DescriptorFlags::ReadOnly)
            : std::nullopt;

    if (auto tensor = arg.value().getType().dyn_cast<RankedTensorType>()) {
      bindings.push_back(IREE::HAL::DescriptorSetBindingAttr::get(
          ctx, /*ordinal=*/bindings.size(),
          IREE::HAL::DescriptorType::StorageBuffer, flags));
    }
  }

  for (auto ret : dispatch.getResults()) {
    // Skip tied results as they alrady passed in as arguments.
    if (dispatch.getTiedResultOperand(ret)) continue;

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

  // Partition arguments types, so that all constants pushed to the end.
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

// Updates Triton dispatch operation according to the arguments permutation
// computed by the `updateTritonFunctionForAbi` function defined above.
static void updateTritonDispatchForAbi(DispatchOp dispatch,
                                       SmallVector<int64_t> permutaion) {
  SmallVector<Value> args = llvm::to_vector(dispatch.getArguments());
  SmallVector<Value> abiArgs(dispatch.getArguments().size());

  // Number of result tensors not tied to any of the operands.
  size_t numUntiedResults = llvm::count_if(
      dispatch.getResults(),
      [&](Value result) { return !dispatch.getTiedResultOperand(result); });

  for (unsigned i = 0; i < args.size(); ++i) {
    int64_t newArgIdx = permutaion[i];

    // In Triton function type scalar (constant) arguments are passed after
    // destination pointers for results, so we have to adjust index for that.
    if (newArgIdx >= args.size()) newArgIdx -= numUntiedResults;

    abiArgs[newArgIdx] = args[i];
  }

  dispatch.getArgumentsMutable().assign(abiArgs);

  // Tie results to the new operands after updating arguments.
  if (auto tiedOperands = dispatch.getTiedOperands()) {
    SmallVector<int64_t> abiTiedOperands;

    for (auto attr : tiedOperands->getValue()) {
      auto idx = attr.cast<IntegerAttr>().getInt();
      abiTiedOperands.push_back(idx < 0 ? idx : permutaion[idx]);
    }

    dispatch.setTiedOperandsAttr(
        OpBuilder(dispatch).getIndexArrayAttr(abiTiedOperands));
  }
}

static void refineTritonAbi(DispatchOp dispatch, SymbolTable& symTable) {
  // Find the Triton export declaration corresponding to dispatch operation.
  auto exportOp = symTable.lookupNearestSymbolFrom<ExecutableExportOp>(
      dispatch, dispatch.getEntryPoint());
  assert(!exportOp.getLayout().has_value() && "layout already defined");

  // Find the exported function in the inner module.
  auto innerModule = exportOp.getParentOp<ExecutableOp>().getInnerModule();
  auto callee = symTable.lookupNearestSymbolFrom<triton::FuncOp>(
      innerModule, exportOp.getFunctionRefAttr());
  assert(callee && "callee must be a Triton function");

  // Update Triton function and a dispatch operation to be ABI compatible.
  if (auto permutation = updateTritonFunctionForAbi(callee)) {
    updateTritonDispatchForAbi(dispatch, *permutation);
  }

  // Set the IREE pipeline layout (aka ABI) inferred from the Triton signature.
  IREE::HAL::PipelineLayoutAttr layout = getPipelineLayout(dispatch, callee);
  exportOp.setLayoutAttr(layout);
}

// TODO(ezhulenev): Currently we assume that there is a 1-to-1 mapping between
// dispatch and executable, however it might not always be the case, and
// different dispatches might use different combination of tied results, and
// each callsite might have a unique ABI.
class RefineTritonAbi : public ::impl::RefineTritonAbiBase<RefineTritonAbi> {
 public:
  void runOnOperation() override {
    SmallVector<DispatchOp> dispatches;
    getOperation()->walk([&](DispatchOp op) { dispatches.push_back(op); });

    // Don't build potentially expensive symbols table if we have no work to do.
    if (dispatches.empty()) return;

    SymbolTable symTable(getOperation());
    for (auto dispatch : dispatches) {
      refineTritonAbi(dispatch, symTable);
    }
  }
};

std::unique_ptr<OperationPass<ModuleOp>> createRefineTritonAbi() {
  return std::make_unique<RefineTritonAbi>();
}

}  // namespace openxla::compiler::nvgpu::tritonflow
