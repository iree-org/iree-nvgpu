// Copyright 2023 The OpenXLA Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef OPENXLA_NVGPU_TRANSFORMS_PASSES_H_
#define OPENXLA_NVGPU_TRANSFORMS_PASSES_H_

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

namespace openxla::compiler::nvgpu::cudnn {

// Converts stable HLO ops to cudnn ops
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createConvertHLOToCUDNNPass();

}  // namespace openxla::compiler::nvgpu::cudnn

#endif  // OPENXLA_NVGPU_TRANSFORMS_PASSES_H_
