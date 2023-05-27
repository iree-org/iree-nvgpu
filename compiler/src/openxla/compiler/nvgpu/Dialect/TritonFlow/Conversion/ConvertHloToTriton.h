// Copyright 2023 The OpenXLA Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef OPENXLA_COMPILER_NVGPU_DIALECT_TRITONFLOW_CONVERSION_CONVERT_HLO_TO_TRITON_H_
#define OPENXLA_COMPILER_NVGPU_DIALECT_TRITONFLOW_CONVERSION_CONVERT_HLO_TO_TRITON_H_

#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"

namespace openxla::compiler::nvgpu::tritonflow {

// Appends conversion patterns from HLO custom calls to TritonFlow dialect call
// operations. We rely on compile time custom calls created by the Jax-Triton.
void populateHloToTritonPatterns(mlir::TypeConverter &typeConverter,
                                 mlir::RewritePatternSet &patterns);

}  // namespace openxla::compiler::nvgpu::tritonflow

#endif  // OPENXLA_COMPILER_NVGPU_DIALECT_TRITONFLOW_CONVERSION_CONVERT_HLO_TO_TRITON_H_
