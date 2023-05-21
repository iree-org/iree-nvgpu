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

//===----------------------------------------------------------------------===//
// cublas.gemm operation
//===----------------------------------------------------------------------===//

def CUBLAS_GemmOp : CUBLAS_Op<"gemm", [
  AttrSizedOperandSegments,
  DeclareOpInterfaceMethods<Util_TiedOpInterface>,
  DeclareOpInterfaceMethods<Util_ShapeAwareOp, [
    "getOperandDynamicDims", "getResultDynamicDims",
  ]>
]> {
  let summary = "general matrix multiply operation";

  let description = [{
    General matrix multiplication operation targeting `cublasGemmEx` cuBLAS API.
  }];

  // TODO(ezhulenev): Instead of variadic arguments and results we should be
  // able to specify separate arguments for `a`, `b` and `c`, but it requires
  // updating `Util_TiedOpInterface` to work with non-variadic arguments and
  // results. Currently we use verifier to check that we have the right number
  // of arguments and results.

  let arguments = (ins
    Variadic<CUBLAS_Tensor2D>:$arguments,
    OptionalAttr<CUBLAS_OperationAttr>:$transa,
    OptionalAttr<CUBLAS_OperationAttr>:$transb,
    OptionalAttr<CUBLAS_GemmAlgoAttr>:$algo,
    OptionalAttr<CUBLAS_DataTypeAttr>:$computeType,
    Variadic<Index>:$argument_dims,
    Variadic<Index>:$result_dims,
    OptionalAttr<Util_TiedOpStorageAttr>:$tied_operands
  );

  let results = (outs
    Variadic<CUBLAS_Tensor2D>:$results
  );

  let assemblyFormat = [{
    `(` $arguments `)`
      (`transa` `=` $transa^)?
      (`transb` `=` $transb^)?
      (`algo` `=` $algo^)?
      (`compute_type` `=` $computeType^)?
        attr-dict `:`
        custom<ShapedFunctionType>(ref($arguments),
                                   type($arguments), $argument_dims,
                                   type($results), $result_dims,
                                   $tied_operands)
  }];

  let extraClassDeclaration = [{
    mlir::Value getA() { return getOperand(0); }
    mlir::Value getB() { return getOperand(1); }
    mlir::Value getC() {
      return getNumOperands() >= 3 ? getOperand(2) : nullptr;
    }
  }];

  let hasVerifier = 1;
}

#endif // CUBLAS_OPS
