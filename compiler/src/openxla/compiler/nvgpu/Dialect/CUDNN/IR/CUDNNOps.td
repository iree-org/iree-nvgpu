//===- CUDNNOps.td - CUDNN dialect ops ---------------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CUDNN_OPS
#define CUDNN_OPS

include "openxla/compiler/nvgpu/Dialect/CUDNN/IR/CUDNNDialect.td"
include "openxla/compiler/nvgpu/Dialect/CUDNN/IR/CUDNNTypes.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/FunctionInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def CUDNN_PointWiseReluOp : CUDNN_Op<"pointwise_relu", [Pure]> {
    let summary = "Pointwise Relu";

    let arguments = (ins
      CUDNN_TensorType:$input,
      TypeAttr:$compute_type,
      F64Attr:$lower_clip
    );
    let results = (outs CUDNN_TensorType:$res);

    let assemblyFormat = [{
      `(` $input `)` `type` `=` $compute_type `lower_clip` `=` $lower_clip
        attr-dict `:` qualified(type($input)) `->` qualified(type($res))
    }];
}

def CUDNN_CrossCorrelationOp : CUDNN_Op<"cross_correlation", [Pure]> {
    let summary = "Cross correlation";

    let arguments = (ins
      CUDNN_TensorDescType:$x,
      CUDNN_TensorDescType:$w,
      TypeAttr:$element_type,
      F32Attr:$alpha,
      F32Attr:$beta,
      I32Attr:$spatial_dim_count,
      DenseI64ArrayAttr:$spatial_stride,
      DenseI64ArrayAttr:$pre_padding,
      DenseI64ArrayAttr:$post_padding,
      DenseI64ArrayAttr:$dilation
    );
    let results = (outs CUDNN_TensorDescType:$y);

    let assemblyFormat = [{
      `(` $x `,` $w `)` `type` `=` $element_type
          `alpha` `=` $alpha
          `beta` `=` $beta
          `spatial_dim_count` `=` $spatial_dim_count
          `spatial_stride` `=`  $spatial_stride
          `pre_padding` `=` $pre_padding
          `post_padding` `=` $post_padding
          `dilation` `=` $dilation
          attr-dict `:` qualified(type($x)) `,` qualified(type($w)) `->` qualified(type($y))
    }];
}

// MatMul
// ------
def CUDNN_MatMulOp : CUDNN_Op<"matmul", [Pure]> {
    let summary = "Matmul";

    let arguments = (ins
      CUDNN_TensorDescType:$a,
      CUDNN_TensorDescType:$b,
      TypeAttr:$element_type
    );
    let results = (outs CUDNN_TensorDescType:$c);

    let assemblyFormat = [{
      `(` $a `,` $b `)` `type` `=` $element_type
         attr-dict `:` qualified(type($a)) `,` qualified(type($b)) `->` qualified(type($c))
    }];
}

// Reduction
// ---------
def CUDNN_ReductionOp : CUDNN_Op<"reduction", [Pure]> {
    let summary = "Reduction";

    let arguments = (ins
      CUDNN_TensorDescType:$x,
      // TODO: Shouldn't be enum.
      I32Attr:$reduction_op,
      TypeAttr:$element_type
    );
    let results = (outs CUDNN_TensorDescType:$y);

    let assemblyFormat = [{
       `(` $x `)` `type` `=` $element_type `reduction_op` `=` $reduction_op
         attr-dict `:` qualified(type($x)) `->` qualified(type($y))
    }];
}

//===----------------------------------------------------------------------===//
// cudnn.add operation
//===----------------------------------------------------------------------===//

def CUDNN_AddOp : CUDNN_Op<"add", [Pure]> {
    let summary = "Add";

    let arguments = (ins
      CUDNN_TensorType:$x,
      CUDNN_TensorType:$b,
      F32Attr:$alpha,
      F32Attr:$alpha2
    );
    let results = (outs CUDNN_TensorType:$y);

    let assemblyFormat = [{
      `(` $x `,` $b `)`
         `alpha` `=` $alpha
         `alpha2` `=` $alpha2
          attr-dict `:`
          functional-type(operands, results)
    }];
}

//===----------------------------------------------------------------------===//
// cudnn.bias operation
//===----------------------------------------------------------------------===//

def CUDNN_BiasOp : CUDNN_Op<"bias", [Pure]> {
    let summary = "Bias";

    let arguments = (ins
      CUDNN_TensorType:$x,
      CUDNN_TensorType:$b
    );
    let results = (outs CUDNN_TensorType:$y);

    let assemblyFormat = [{
      `(` $x `,` $b `)`
          attr-dict `:`
          functional-type(operands, results)
    }];
}

//===----------------------------------------------------------------------===//
// cudnn.convolution operation
//===----------------------------------------------------------------------===//

def CUDNN_ConvolutionOp : CUDNN_Op<"convolution", [Pure]> {
    let summary = "Convolution";

    let arguments = (ins
      CUDNN_TensorType:$x,
      CUDNN_TensorType:$w,
      F32Attr:$alpha,
      F32Attr:$beta,
      I32Attr:$spatial_dim_count,
      DenseI64ArrayAttr:$spatial_stride,
      DenseI64ArrayAttr:$pre_padding,
      DenseI64ArrayAttr:$post_padding,
      DenseI64ArrayAttr:$dilation
    );

    let results = (outs CUDNN_TensorType:$y);

    let assemblyFormat = [{
      `(` $x `,` $w `)`
         `alpha` `=` $alpha
         `beta` `=` $beta
         `spatial_dim_count` `=` $spatial_dim_count
         `spatial_stride` `=`  $spatial_stride
         `pre_padding` `=` $pre_padding
         `post_padding` `=` $post_padding
         `dilation` `=` $dilation
         attr-dict `:`
         functional-type(operands, results)
    }];

    // TODO(ezhulenev): Add a custom verifier to check for shape errors.
}

//===----------------------------------------------------------------------===//
// cudnn.graph operation
//===----------------------------------------------------------------------===//

def CUDNN_GraphOp : CUDNN_Op<"graph", [
    IsolatedFromAbove,
    FunctionOpInterface,
    CallableOpInterface,
    Symbol,
    SingleBlockImplicitTerminator<"openxla::compiler::nvgpu::cudnn::ReturnOp">
  ]> {
  let summary = "operation defining cuDNN graph building function";

  let description = [{
    Represents a function building a cuDNN graph from a cuDNN operations. At
    run time this operation lowered to vm function calling into cuDNN module API
    to construct a cuDNN operation graph object.
  }];

  let arguments = (ins
    TypeAttrOf<FunctionType>:$function_type,
    OptionalAttr<DictArrayAttr>:$arg_attrs,
    OptionalAttr<DictArrayAttr>:$res_attrs
  );

  let regions = (region AnyRegion:$body);

  let extraClassDeclaration = [{
    /// Returns the argument types of cuDNN graph.
    llvm::ArrayRef<mlir::Type> getArgumentTypes() {
      return getFunctionType().getInputs();
    }

    /// Returns the result types of this cuDNN graph.
    llvm::ArrayRef<mlir::Type> getResultTypes() {
      return getFunctionType().getResults();
    }

    /// Hook for OpTrait::FunctionLike, called after verifying that the 'type'
    /// attribute is present. This can check for preconditions of the
    /// getNumArguments hook not failing.
    mlir::LogicalResult verifyType();

    /// CallableOpInterface implementation.
    mlir::Region *getCallableRegion() { return &getBody(); }
    llvm::ArrayRef<mlir::Type> getCallableResults() {
      return getFunctionType().getResults();
    }
    mlir::ArrayAttr getCallableArgAttrs() {
      return getArgAttrs().value_or(nullptr);
    }
    mlir::ArrayAttr getCallableResAttrs() {
      return getResAttrs().value_or(nullptr);
    }
  }];

  let hasCustomAssemblyFormat = 1;
}

//===----------------------------------------------------------------------===//
// cudnn.return operation
//===----------------------------------------------------------------------===//

def CUDNN_ReturnOp : CUDNN_Op<"return", [
    Pure,
    ReturnLike,
    Terminator,
    HasParent<"::openxla::compiler::nvgpu::cudnn::GraphOp">
  ]> {
  let summary = "return operation";
  let description = [{
    Represents a return operation within a cuDNN graph.

    ```
    cudnn.graph @foo(%arg: !cudnn.tensor<?x?x?xf32>)
                        -> !cudnn.tensor<?x?x?xf32> {
      %0 = ... : !cudnn.tensor<?x?x?xf32>
      return %0: !cudnn.tensor<?x?x?xf32>
    }
    ```
  }];

  let arguments = (ins
    Variadic<CUDNN_TensorType>:$operands
  );

  let assemblyFormat = [{
    attr-dict ($operands^ `:` type($operands))?
  }];

  let builders = [
    OpBuilder<(ins),
    [{
      build($_builder, $_state, std::nullopt);
    }]>,
  ];
}

//===----------------------------------------------------------------------===//
// cudnn.call operation
//===----------------------------------------------------------------------===//

def CUDNN_CallOp : CUDNN_Op<"call", [
    CallOpInterface,
    DeclareOpInterfaceMethods<SymbolUserOpInterface>
  ]> {
  let summary = "cuDNN call operation";

  let description = [{
    Calls a cuDNN graph with the given arguments.
  }];

  let arguments = (ins
    CUDNN_GraphRefAttr:$callee,
    Variadic<AnyRankedTensor>:$arguments
  );

  let results = (outs
    Variadic<AnyRankedTensor>:$results
  );

  let assemblyFormat = [{
    $callee `(` operands `)` attr-dict `:` functional-type(operands, results)
  }];

  let extraClassDeclaration = [{
    /// Get the argument operands to the called function.
    operand_range getArgOperands() {
      return {arg_operand_begin(), arg_operand_end()};
    }
    operand_iterator arg_operand_begin() { return operand_begin(); }
    operand_iterator arg_operand_end() { return operand_end(); }

    /// Return the callee of this operation.
    mlir::CallInterfaceCallable getCallableForCallee() {
      return getOperation()->getAttrOfType<mlir::FlatSymbolRefAttr>("callee");
    }
  }];
}

#endif // CUDNN_OPS
