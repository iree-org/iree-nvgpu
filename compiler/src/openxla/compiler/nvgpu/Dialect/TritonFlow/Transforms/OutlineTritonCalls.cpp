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

#include "llvm/IR/Module.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Support/LogicalResult.h"
#include "openxla/compiler/nvgpu/Dialect/TritonFlow/Transforms/PassDetail.h"
#include "openxla/compiler/nvgpu/Dialect/TritonFlow/Transforms/Passes.h"

#define GEN_PASS_DEF_OUTLINETRITONCALLS
#include "openxla/compiler/nvgpu/Dialect/TritonFlow/Transforms/Passes.h.inc"

using namespace mlir;

namespace openxla::compiler::nvgpu::tritonflow {

static LogicalResult outlineTritonFlowCallOp(tritonflow::CallOp call,
                                             SymbolTable& symTable) {
  // Get the Triton function callee.
  auto callee = symTable.lookup<triton::FuncOp>(call.getCallee());
  ImplicitLocOpBuilder b(callee->getLoc(), call->getParentOfType<ModuleOp>());

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
  auto exportExecutableOp =
      b.create<ExecutableExportOp>(callee.getSymName(), call.getCalleeAttr());

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
