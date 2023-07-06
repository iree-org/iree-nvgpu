// Copyright 2023 The OpenXLA Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CUBLAS_DIALECT
#define CUBLAS_DIALECT

include "mlir/IR/OpBase.td"

def CUBLAS_Dialect : Dialect {
  let name = "cublas";

  let cppNamespace = "::openxla::compiler::nvgpu::cublas";

  let description = [{
    OpenXLA CUBLAS dialect provides operations for integrating NVIDIA cuBLAS
    library into the OpenXLA compiler.
  }];

  let useDefaultTypePrinterParser = 1;
  let useDefaultAttributePrinterParser = 1;

  let extraClassDeclaration = [{
    void registerAttrs();
    void registerTypes();
  }];
}

//===----------------------------------------------------------------------===//
// cuBLAS types and helpers
//===----------------------------------------------------------------------===//

// TODO(ezhulenev): Add all supported data types.
def CUBLAS_AnyDataType : AnyTypeOf<[F32,F64], "any supported cuBLAS data type">;

def CUBLAS_Tensor2D : 2DTensorOf<[CUBLAS_AnyDataType]>;

//===----------------------------------------------------------------------===//
// Base operation definition
//===----------------------------------------------------------------------===//

class CUBLAS_Op<string mnemonic, list<Trait> traits = []> :
        Op<CUBLAS_Dialect, mnemonic, traits>;

#endif // CUBLAS_DIALECT
