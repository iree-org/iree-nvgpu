// Copyright 2023 The OpenXLA Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef TRITON_DIALECT
#define TRITON_DIALECT

include "mlir/IR/OpBase.td"

def TritonFlow_Dialect : Dialect {
  let name = "triton";
  let cppNamespace = "::openxla::compiler::nvgpu::tritonflow";
  let description = [{
    OpenXLA Triton dialect contains operations glueing together Triton compiler
    and IREE runtime and allows embedding Triton programs into OpenXLA modules.
  }];
}

def TritonFlow_Dim : TypeAlias<Index>;

//===----------------------------------------------------------------------===//
// General types and helpers
//===----------------------------------------------------------------------===//

class TritonFlow_SymbolRefAttr :
        Attr<CPred<"$_self.isa<mlir::FlatSymbolRefAttr>()">,
             "symbol reference attribute"> {
  let storageType = [{ mlir::FlatSymbolRefAttr }];
  let returnType = [{ mlir::StringRef }];
  let valueType = NoneType;
  let constBuilderCall = "mlir::SymbolRefAttr::get($_builder.getContext(), $0)";
}

def TritonFlow_FuncRefAttr : TritonFlow_SymbolRefAttr;

//===----------------------------------------------------------------------===//
// Base operation definition
//===----------------------------------------------------------------------===//

class TritonFlow_Op<string mnemonic, list<Trait> traits = []> :
        Op<TritonFlow_Dialect, mnemonic, traits>;

#endif // TRITON_DIALECT
