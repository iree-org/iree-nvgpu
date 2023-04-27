// Copyright 2023 The OpenXLA Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <algorithm>
#include <numeric>

#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "openxla/compiler/nvgpu/Dialect/CUDNN/IR/CUDNNDialect.h"
#include "openxla/compiler/nvgpu/Transforms/Passes.h"
#include "openxla/compiler/nvgpu/Conversion/ConvertHLOToCUDNN.h"

#define GEN_PASS_DEF_CONVERTHLOTOCUDNN
#include "openxla/compiler/nvgpu/Transforms/Passes.h.inc"

using namespace mlir;

namespace openxla::compiler::nvgpu::cudnn {

namespace {

class ConvertHLOToCUDNN : public ::impl::ConvertHLOToCUDNNBase<
    ConvertHLOToCUDNN> {
 public:
  void runOnOperation() override {
    auto apply = [&](auto populatePatterns) {
      RewritePatternSet patterns(&getContext());
      (*populatePatterns)(patterns);
      return applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
    };
    if (failed(apply(populateOutlineHLOToCUDNNPatterns)))
      return signalPassFailure();
    if (failed(apply(populateConvertHLOToCUDNNPatterns)))
      return signalPassFailure();
  }
};

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>> createConvertHLOToCUDNNPass() {
  return std::make_unique<ConvertHLOToCUDNN>();
}

}  // namespace openxla::compiler::nvgpu::cudnn
