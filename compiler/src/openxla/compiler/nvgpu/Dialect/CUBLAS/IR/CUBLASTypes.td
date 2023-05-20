// Copyright 2023 The OpenXLA Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CUBLAS_TYPES
#define CUBLAS_TYPES

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"
include "openxla/compiler/nvgpu/Dialect/CUBLAS/IR/CUBLASDialect.td"

//===----------------------------------------------------------------------===//
// cuBLAS type definitions
//===----------------------------------------------------------------------===//

class CUBLAS_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<CUBLAS_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def CUBLAS_HandleType : CUBLAS_Type<"Handle", "handle"> {
    let summary = "CUBLAS handle";
}

//===----------------------------------------------------------------------===//
// cuBLAS data type corresponding to cudaDataType
//===----------------------------------------------------------------------===//

// TODO(ezhulenev): This really belogs to CUDA dialect, but we keep it as a part
// of cuBLAS library integration for now because we do not have any other users.

def CUBLAS_DATATYPE_R_32F : I32EnumAttrCase<"CUDA_R_32F", 0>;

def CUBLAS_DataType : I32EnumAttr<"DataType",
    "cuBLAS data type",
    [
      CUBLAS_DATATYPE_R_32F,
    ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::openxla::compiler::nvgpu::cublas";
}

#endif // CUBLAS_TYPES
