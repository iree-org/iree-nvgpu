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
// cuBLAS operation type corresponding to cublasOperation_t
//===----------------------------------------------------------------------===//

def CUBLAS_OPERATION_OP_N : I32EnumAttrCase<"N", 0>;
def CUBLAS_OPERATION_OP_T : I32EnumAttrCase<"T", 1>;
def CUBLAS_OPERATION_OP_C : I32EnumAttrCase<"C", 2>;

def CUBLAS_Operation : I32EnumAttr<"CublasOperation",
    "cuBLAS operation",
    [
      CUBLAS_OPERATION_OP_N,
      CUBLAS_OPERATION_OP_T,
      CUBLAS_OPERATION_OP_C
    ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::openxla::compiler::nvgpu::cublas";
}

def CUBLAS_OperationAttr :
  EnumAttr<CUBLAS_Dialect, CUBLAS_Operation, "operation">;

//===----------------------------------------------------------------------===//
// cuBLAS data type corresponding to cudaDataType
//===----------------------------------------------------------------------===//

// TODO(ezhulenev): This really belogs to CUDA dialect, but we keep it as a part
// of cuBLAS library integration for now because we do not have any other users.

def CUBLAS_DATATYPE_R_32F : I32EnumAttrCase<"R_32F", 0>;
def CUBLAS_DATATYPE_R_64F : I32EnumAttrCase<"R_64F", 1>;

def CUBLAS_DataType : I32EnumAttr<"DataType",
    "cuBLAS data type",
    [
      CUBLAS_DATATYPE_R_32F,
      CUBLAS_DATATYPE_R_64F,
    ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::openxla::compiler::nvgpu::cublas";
}

def CUBLAS_DataTypeAttr :
  EnumAttr<CUBLAS_Dialect, CUBLAS_DataType, "data_type">;

//===----------------------------------------------------------------------===//
// cuBLAS gemm algorithm type corresponding to cublasGemmAlgo_t
//===----------------------------------------------------------------------===//

// TODO(ezhulenev): Add all supported GEMM algorithms.

def CUBLAS_GEMMALGO_ALGO0 : I32EnumAttrCase<"ALGO0", 0>;

def CUBLAS_GemmAlgo : I32EnumAttr<"CublasGemmAlgo",
    "cuBLAS gemm algorithm",
    [
      CUBLAS_GEMMALGO_ALGO0,
    ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::openxla::compiler::nvgpu::cublas";
}

def CUBLAS_GemmAlgoAttr :
  EnumAttr<CUBLAS_Dialect, CUBLAS_GemmAlgo, "algo">;


#endif // CUBLAS_TYPES
