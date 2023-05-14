// Copyright 2023 The OpenXLA Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef TRITON_OPS
#define TRITON_OPS

include "mlir/IR/OpBase.td"
include "mlir/IR/SymbolInterfaces.td"

include "openxla/compiler/nvgpu/Dialect/TritonFlow/IR/TritonFlowDialect.td"

//===----------------------------------------------------------------------===//
// triton.dispatch operation
//===----------------------------------------------------------------------===//

// TODO(ezhulenev): We should follow Flow dialect design and introduce Triton
// executable to carry Triton compilation options. For now we just dispatch
// a Triton function and hardcode all other parameters.

def TritonFlow_DispatchOp : TritonFlow_Op<"dispatch", [
    AttrSizedOperandSegments,
    DeclareOpInterfaceMethods<SymbolUserOpInterface>
  ]> {
  let summary = "Triton dispatch operation";

  let description = [{
    Dispatches a Triton function with the given arguments.
  }];

  let arguments = (ins
    Variadic<TritonFlow_Dim>:$grid,
    TritonFlow_FuncRefAttr:$callee,
    Variadic<AnyType>:$arguments
  );

  let results = (outs
    Variadic<AnyType>:$results
  );

  let assemblyFormat = [{
    $callee
      `[` $grid `]` ``
      `(` $arguments `)`
       attr-dict `:`
       functional-type($arguments, $results)
  }];
}

#endif // TRITON_OPS
