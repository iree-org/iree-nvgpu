// Copyright 2023 The OpenXLA Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef OPENXLA_COMPILER_NVGPU_DIALECT_FLOWTRANSFORMEXTENSION
#define OPENXLA_COMPILER_NVGPU_DIALECT_FLOWTRANSFORMEXTENSION

include "mlir/Dialect/Transform/IR/TransformDialect.td"
include "mlir/Dialect/Transform/IR/TransformInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"

def FilterOutAlreadyInDispatchRegionOp : Op<
    Transform_Dialect, "iree.filter_out_already_in_dispatch_region", [
      DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
      DeclareOpInterfaceMethods<TransformOpInterface>]> {
  let summary = 
    "empties the handles if payloads are nested in a dispatch region";
  let description = [{
    If any of the operand handles are associated with payload operations that
    are already included in a dispatch region, returns empty handles. Otherwise,
    returns new handles associated with the same payloads as the operand
    handles. Always has the same number of result handles as the operand
    handles. Never produces failures.
  }];

  let arguments = (ins Variadic<TransformHandleTypeInterface>:$target);
  let results = (outs Variadic<TransformHandleTypeInterface>:$transformed);
  let assemblyFormat =
      [{ $target attr-dict `:` functional-type(operands, results) }];

  let hasVerifier = 1;
  let cppNamespace = "mlir::iree_compiler::IREE::transform_dialect";
}

def RegisterNVGPUMatchCallbacks : Op<
    Transform_Dialect, "iree.register_nvgpu_match_callbacks", [
      DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
      DeclareOpInterfaceMethods<TransformOpInterface>]> {
  let assemblyFormat = "attr-dict";
  let cppNamespace = "mlir::iree_compiler::IREE::transform_dialect";
}

#endif  // OPENXLA_COMPILER_NVGPU_DIALECT_FLOWTRANSFORMEXTENSION
