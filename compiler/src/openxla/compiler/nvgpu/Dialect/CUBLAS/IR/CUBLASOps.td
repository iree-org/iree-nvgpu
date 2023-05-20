// Copyright 2023 The OpenXLA Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CUBLAS_OPS
#define CUBLAS_OPS

include "mlir/IR/OpBase.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

include "iree/compiler/Dialect/HAL/IR/HALBase.td"
include "openxla/compiler/nvgpu/Dialect/CUBLAS/IR/CUBLASDialect.td"
include "openxla/compiler/nvgpu/Dialect/CUBLAS/IR/CUBLASTypes.td"

//===----------------------------------------------------------------------===//
// cublas.handle operation
//===----------------------------------------------------------------------===//

def CUBLAS_HandleOp : CUBLAS_Op<"handle", [Pure]> {
  let summary = "Returns a cuBLAS handle for the given device";

  let arguments = (ins
    HAL_Device:$device
  );

  let results = (outs
    CUBLAS_HandleType:$result
  );

  let assemblyFormat = [{
    `(` $device `)` attr-dict `:` type($result)
  }];

  let builders = [
    OpBuilder<(ins),
    [{
      $_state.addTypes({HandleType::get($_builder.getContext())});
    }]>,
  ];
}

#endif // CUBLAS_OPS
