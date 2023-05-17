// Copyright 2023 The OpenXLA Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef OPENXLA_COMPILER_NVGPU_DIALECT_CUDNN_TRANSFORMS_PASSES_H_
#define OPENXLA_COMPILER_NVGPU_DIALECT_CUDNN_TRANSFORMS_PASSES_H_

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

namespace openxla::compiler::nvgpu::cudnn {

//===----------------------------------------------------------------------===//
// Transformations on cuDNN dialect
//===----------------------------------------------------------------------===//

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createExpandCudnnOperationsPass();

//===----------------------------------------------------------------------===//
// Conversion from cuDNN dialect
//===----------------------------------------------------------------------===//

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createConvertCudnnToRuntimePass();

//===----------------------------------------------------------------------===//
// Normalize stable HLO convolution layouts
//===----------------------------------------------------------------------===//

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
createNormalizeHLOConvolutionLayoutsPass();

}  // namespace openxla::compiler::nvgpu::cudnn

#endif  // OPENXLA_COMPILER_NVGPU_DIALECT_CUDNN_TRANSFORMS_PASSES_H_
