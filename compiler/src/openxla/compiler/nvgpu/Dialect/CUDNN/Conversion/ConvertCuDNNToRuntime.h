// Copyright 2023 The OpenXLA Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef OPENXLA_COMPILER_NVGPU_DIALECT_CUDNN_CONVERSION_CONVERT_CUDNN_TO_RUNTIME_H_
#define OPENXLA_COMPILER_NVGPU_DIALECT_CUDNN_CONVERSION_CONVERT_CUDNN_TO_RUNTIME_H_

#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

namespace openxla::compiler::nvgpu {

// Appends cuDNN dialect to cudNN runtime patterns to the given pattern list.
// Conversion patterns lower from cuDNN dialect operations to function calls
// corresponding to the cuDNN runtime (implemented as a custom VM module).
void populateCuDNNToRuntimePatterns(mlir::MLIRContext *context,
                                    mlir::TypeConverter &typeConverter,
                                    mlir::RewritePatternSet &patterns);

}  // namespace openxla::compiler::nvgpu

#endif  // OPENXLA_COMPILER_NVGPU_DIALECT_CUDNN_CONVERSION_CONVERT_CUDNN_TO_RUNTIME_H_
