// Copyright 2023 The OpenXLA Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef OPENXLA_COMPILER_NVGPU_CONVERSION_CONVERT_HLO_TO_CUDNN_H_
#define OPENXLA_COMPILER_NVGPU_CONVERSION_CONVERT_HLO_TO_CUDNN_H_

#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

namespace openxla::compiler::nvgpu::cudnn {

// Appends StableHLO to cudNN dialect patterns to the given pattern list.
void populateHLOToCUDNNPatterns(mlir::RewritePatternSet &patterns);

}  // namespace openxla::compiler::nvgpu::cudnn

#endif  // OPENXLA_COMPILER_NVGPU_CONVERSION_CONVERT_HLO_TO_CUDNN_H_