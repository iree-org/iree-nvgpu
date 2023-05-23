// Copyright 2023 The OpenXLA Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef TRITONFLOW_DIALECT
#define TRITONFLOW_DIALECT

include "mlir/IR/OpBase.td"

def TritonFlow_Dialect : Dialect {
  let name = "triton";
  let cppNamespace = "::openxla::compiler::nvgpu::tritonflow";
  let description = [{
    OpenXLA Triton dialect contains operations glueing together Triton compiler
    and IREE runtime and allows embedding Triton programs into OpenXLA modules.
  }];
}

//===----------------------------------------------------------------------===//
// Base operation definition
//===----------------------------------------------------------------------===//

class TritonFlow_Op<string mnemonic, list<Trait> traits = []> :
        Op<TritonFlow_Dialect, mnemonic, traits>;

#endif // TRITONFLOW_DIALECT
