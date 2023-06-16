// Copyright 2023 The OpenXLA Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef OPENXLA_COMPILER_NVGPU_DIALECT_TRITONFLOW_CONVERSION_CONVERT_TRITON_TO_FLOW_DISPATCH_H_
#define OPENXLA_COMPILER_NVGPU_DIALECT_TRITONFLOW_CONVERSION_CONVERT_TRITON_TO_FLOW_DISPATCH_H_

#include "iree/compiler/Utils/OptionUtils.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"

namespace openxla::compiler::nvgpu::tritonflow {

// TODO(ezhulenev): TritonOptions should be added to the Triton plugin
// registration (see PluginRegistration.cpp).
struct TritonOptions {
  // TODO(ezhulenev): This is a very old compute capability version that happens
  // to work on P100 GPUs attached to ezhulenev@ desktop. We have to target 7.0+
  // as Triton officially does not support lower versions.
  int32_t compute_capability = 60;

  // TODO(ezhulenev): We have to get PTX version from the CUDA Toolkit version.
  // See example in Triton compiler.py `ptx_get_version`.
  int32_t ptx_version = 60;

  // TODO(ezhulenev): Shouldn't warps and stages be configurable per-kernel?
  // Anyway here we again hardcode them to get minimal proof of concept running
  // end to end.
  int32_t num_warps = 4;
  int32_t num_stages = 1;

  void bindOptions(mlir::iree_compiler::OptionsBinder &binder);
  using FromFlags = mlir::iree_compiler::OptionsFromFlags<TritonOptions>;
};

// Appends OpenXLA Triton dialect to OpenXLA Triton runtime patterns to the
// given pattern list. OpenXLA Triton dialect contains operations glueing
// together Triton compiler and IREE runtime. Triton compiler uses its own
// triton dialects (ttir and ttgir).
void populateTritonToFlowDispatchPatterns(mlir::TypeConverter &typeConverter,
                                          mlir::RewritePatternSet &patterns);

// Build a compilation pipeline that lowers from Triton IR to LLVM. This is an
// implementation details of the Triton to OpenXLA Triton runtime lowering and
// exposed only for testing it in isolation.
void buildTritonCompilationPipeline(mlir::OpPassManager &pm,
                                    const TritonOptions &opts);

}  // namespace openxla::compiler::nvgpu::tritonflow

#endif  // OPENXLA_COMPILER_NVGPU_DIALECT_TRITONFLOW_CONVERSION_CONVERT_TRITON_TO_FLOW_DISPATCH_H_
