// Copyright 2023 The OpenXLA Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef TRITONFLOW_OPS
#define TRITONFLOW_OPS

include "mlir/IR/OpBase.td"
include "mlir/IR/SymbolInterfaces.td"

include "iree/compiler/Dialect/Util/IR/UtilBase.td"
include "iree/compiler/Dialect/Util/IR/UtilInterfaces.td"
include "openxla/compiler/nvgpu/Dialect/TritonFlow/IR/TritonFlowDialect.td"

//===----------------------------------------------------------------------===//
// triton.executable operation
//===----------------------------------------------------------------------===//

def TritonFlow_ExecutableOp : TritonFlow_Op<"executable", [
    IsolatedFromAbove,
    SingleBlockImplicitTerminator<"ExecutableEndOp">,
    NativeOpTrait<"SymbolTable">,
    Symbol
]> {
  let summary = "Triton executable module";

  let description = [{
    A Triton executable module containing one or more public Triton functions.
    The contents of the functions are safe to dispatch and can be lowered
    further to target-specific backend IR representation using Triton compiler
    passes. This is very similar to the `flow.executable` operation but
    specialized for the Triton compiler, while the `flow.executable` is a part
    of builtin IREE compilation pipeline.
  }];

  let arguments = (ins
    OptionalAttr<StrAttr>:$sym_visibility,
    SymbolNameAttr:$sym_name
  );

  let regions = (region SizedRegion<1>:$body);

  let assemblyFormat = [{
    custom<SymbolVisibility>($sym_visibility)
    $sym_name
    attr-dict-with-keyword
    regions
  }];

  let extraClassDeclaration = [{
    ::mlir::Block& getBlock() { return getBody().front(); }
    ::mlir::ModuleOp getInnerModule() {
      return *getBlock().getOps<::mlir::ModuleOp>().begin();
    }
  }];

  let hasVerifier = 1;
}

def TritonFlow_ExecutableEndOp : TritonFlow_Op<"executable_end", [
  HasParent<"ExecutableOp">,
  Terminator,
]> {
  let summary = "terminator pseudo-op for the executable op";
  let assemblyFormat = "attr-dict";
}

//===----------------------------------------------------------------------===//
// triton.executable.export operation
//===----------------------------------------------------------------------===//

// TODO(ezhulenev): We might use this mechanism for exporting the same Triton
// function with different Triton compiler options (num_stages, num_warps), or
// specializing for constant values (if/when tl.constexpr will be available in
// Triton MLIR dialect), and custom dispatch grid calculation. Currently it
// only exports a function from the nested module (maybe with an alias).

// TODO(ezhulenev): Should we have a pipeline layout attribute here? Or infer it
// automatically later once lowering to HAL executable? Consider adding optional
// layout, it might be easier to infer it early while the inner module is still
// at Triton level, and don't wait for lowering to LLVM.

def TritonFlow_ExecutableExportOp : TritonFlow_Op<"executable.export", [
  HasParent<"ExecutableOp">,
  Symbol,
  IsolatedFromAbove,
  DeclareOpInterfaceMethods<SymbolUserOpInterface>
]> {
  let summary = "defines a Triton executable entry point for dispatches";

  let description = [{
    Specifies an exported function with an externally-visible alias. Multiple
    exports can reference the same internal function.
  }];

  let arguments = (ins
    OptionalAttr<StrAttr>:$sym_visibility,
    SymbolNameAttr:$sym_name,
    FlatSymbolRefAttr:$function_ref
  );

  let assemblyFormat = [{
    custom<SymbolVisibility>($sym_visibility)
    custom<SymbolAlias>($sym_name, $function_ref)
    attr-dict-with-keyword
  }];
}

//===----------------------------------------------------------------------===//
// triton.dispatch operation
//===----------------------------------------------------------------------===//

// TODO(ezhulenev): Add support for dynamic shapes, so that we can eventually
// lower to `flow.dispatch` with dynamic shapes.

// TODO(ezhulenev): Do we need a function to convert workload into a 3D
// workgroup (grid) similar to `flow.dispatch`? Current assumption is that
// the end user provides grid dimension as a part of `triton.call`. We need to
// revisit it later once we have more clear requirements.

def TritonFlow_DispatchOp : TritonFlow_Op<"dispatch", [
  AttrSizedOperandSegments,
  DeclareOpInterfaceMethods<SymbolUserOpInterface>,
  DeclareOpInterfaceMethods<Util_TiedOpInterface, [
    "getTiedOperandsIndexAndLength",
  ]>,
  DeclareOpInterfaceMethods<Util_ShapeAwareOp, [
    "getOperandDynamicDims", "getResultDynamicDims",
  ]>
]> {
  let summary = "a dispatch of Triton executable across a grid";

  let description = [{
    Dispatches Triton executable across a grid.
  }];

  let arguments = (ins
    Variadic<Index>:$grid,
    SymbolRefAttr:$entry_point,
    Variadic<AnyType>:$arguments,
    Variadic<Index>:$argument_dims,
    Variadic<Index>:$result_dims,
    OptionalAttr<Util_TiedOpStorageAttr>:$tied_operands
  );

  let results = (outs
    Variadic<AnyType>:$results
  );

  let assemblyFormat = [{
    $entry_point
      `[` $grid `]` ``
      `(` $arguments `)`
       attr-dict `:`
       custom<ShapedFunctionType>(ref($arguments),
                                  type($arguments), $argument_dims,
                                  type($results), $result_dims,
                                  $tied_operands)
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// triton.call operation
//===----------------------------------------------------------------------===//

// TODO(ezhulenev): Add support for dynamic shapes, so that we can eventually
// lower to `flow.dispatch` with dynamic shapes.

def TritonFlow_CallOp : TritonFlow_Op<"call", [
  AttrSizedOperandSegments,
  DeclareOpInterfaceMethods<SymbolUserOpInterface>,
  DeclareOpInterfaceMethods<Util_TiedOpInterface, [
    "getTiedOperandsIndexAndLength",
  ]>,
  DeclareOpInterfaceMethods<Util_ShapeAwareOp, [
    "getOperandDynamicDims", "getResultDynamicDims",
  ]>
]> {
  let summary = "Triton call operation";

  let description = [{
    Calls a Triton function with the given arguments.
  }];

  let arguments = (ins
    Variadic<Index>:$grid,
    FlatSymbolRefAttr:$callee,
    Variadic<AnyType>:$arguments,
    Variadic<Index>:$argument_dims,
    Variadic<Index>:$result_dims,
    OptionalAttr<Util_TiedOpStorageAttr>:$tied_operands
  );

  let results = (outs
    Variadic<AnyType>:$results
  );

  let assemblyFormat = [{
    $callee
      `[` $grid `]` ``
      `(` $arguments `)`
       attr-dict `:`
       custom<ShapedFunctionType>(ref($arguments),
                                  type($arguments), $argument_dims,
                                  type($results), $result_dims,
                                  $tied_operands)
  }];

  let hasVerifier = 1;
}

#endif // TRITONFLOW_OPS
