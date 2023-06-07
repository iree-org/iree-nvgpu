// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef OPENXLA_COMPILER_NVGPU_DIALECT_FLOWTRANSFORMEXTENSION_IR_FLOWTRANSFORMEXTENSION_H
#define OPENXLA_COMPILER_NVGPU_DIALECT_FLOWTRANSFORMEXTENSION_IR_FLOWTRANSFORMEXTENSION_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/IR/TransformInterfaces.h"

#define GET_OP_CLASSES
#include "openxla/compiler/nvgpu/Dialect/FlowTransformExtension/IR/FlowTransformExtension.h.inc"

namespace mlir {
class DialectRegistry;

namespace openxla {
namespace nvgpu {
/// Registers the Transform dialect extension with transforms operating on the
/// Flow level in the given registry.
void registerFlowTransformExtension(DialectRegistry &registry);
}  // namespace nvgpu
}  // namespace openxla
}  // namespace mlir

#endif  // OPENXLA_COMPILER_NVGPU_DIALECT_FLOWTRANSFORMEXTENSION_IR_FLOWTRANSFORMEXTENSION_H
