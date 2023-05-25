//===- CUDNNOps.td - CUDNN dialect ops ---------------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CUDNN_OPS
#define CUDNN_OPS

include "iree/compiler/Dialect/HAL/IR/HALBase.td"
include "openxla/compiler/nvgpu/Dialect/CUDNN/IR/CUDNNDialect.td"
include "openxla/compiler/nvgpu/Dialect/CUDNN/IR/CUDNNTypes.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/FunctionInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// cudnn.relu operation
//===----------------------------------------------------------------------===//

def CUDNN_ReluOp : CUDNN_Op<"relu", [Pure]> {
    let summary = "Relu";

    let arguments = (ins
      CUDNN_TensorType:$input,
      TypeAttr:$compute_type,
      F64Attr:$lower_clip
    );
    let results = (outs CUDNN_TensorType:$res);

    let assemblyFormat = [{
      `(` $input `)`
         `type` `=` $compute_type
         `lower_clip` `=` $lower_clip
          attr-dict `:`
          functional-type(operands, results)
    }];
}

//===----------------------------------------------------------------------===//
// cuDNN pointwise unary operations
//===----------------------------------------------------------------------===//

class CUDNN_UnaryOp<string mnemonic, list<Trait> traits = []> :
    CUDNN_Op<mnemonic, !listconcat([Pure], traits)>,
    Arguments<(ins
      CUDNN_TensorType:$x,
      DefaultValuedOptionalAttr<F32Attr, "1.0f">:$alpha)>,
    Results<(outs CUDNN_TensorType:$y)> {
  let assemblyFormat = [{
    `(` $x `)`
      (`alpha` `=` $alpha^)?
        attr-dict `:`
        functional-type(operands, results)
  }];
}

//===----------------------------------------------------------------------===//
// cudnn.sqrt operation
//===----------------------------------------------------------------------===//

def CUDNN_SqrtOp : CUDNN_UnaryOp<"sqrt"> {
  let summary = "Sqrt";
}

//===----------------------------------------------------------------------===//
// cuDNN pointwise binary operations
//===----------------------------------------------------------------------===//

class CUDNN_BinaryOp<string mnemonic, list<Trait> traits = []> :
    CUDNN_Op<mnemonic, !listconcat([Pure], traits)>,
    Arguments<(ins
      CUDNN_TensorType:$x,
      CUDNN_TensorType:$b,
      DefaultValuedOptionalAttr<F32Attr, "1.0f">:$alpha,
      DefaultValuedOptionalAttr<F32Attr, "1.0f">:$alpha2)>,
    Results<(outs CUDNN_TensorType:$y)> {
  let assemblyFormat = [{
    `(` $x `,` $b `)`
      (`alpha` `=` $alpha^)?
      (`alpha2` `=` $alpha2^)?
        attr-dict `:`
        functional-type(operands, results)
  }];
}

//===----------------------------------------------------------------------===//
// cudnn.add operation
//===----------------------------------------------------------------------===//

def CUDNN_AddOp : CUDNN_BinaryOp<"add"> {
  let summary = "Add";
}

//===----------------------------------------------------------------------===//
// cudnn.div operation
//===----------------------------------------------------------------------===//

def CUDNN_DivOp : CUDNN_BinaryOp<"div"> {
  let summary = "Div";
}

//===----------------------------------------------------------------------===//
// cudnn.max operation
//===----------------------------------------------------------------------===//

def CUDNN_MaxOp : CUDNN_BinaryOp<"max"> {
  let summary = "Max";
}

//===----------------------------------------------------------------------===//
// cudnn.mul operation
//===----------------------------------------------------------------------===//

def CUDNN_MulOp : CUDNN_BinaryOp<"mul"> {
  let summary = "Mul";
}

//===----------------------------------------------------------------------===//
// cudnn.sub operation
//===----------------------------------------------------------------------===//

def CUDNN_SubOp : CUDNN_BinaryOp<"sub"> {
  let summary = "Sub";
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
      DenseI64ArrayAttr:$dilation,
      DefaultValuedAttr<CUDNN_ConvolutionModeAttr,
        "ConvolutionMode::CROSS_CORRELATION">:$mode
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
         (`mode` `=` $mode^)?
         attr-dict `:`
         functional-type(operands, results)
    }];

    // TODO(ezhulenev): Add a custom verifier to check for shape errors.
}

//===----------------------------------------------------------------------===//
// cudnn.batch_norm_inference operation
//===----------------------------------------------------------------------===//

def CUDNN_BatchNormInferenceOp : CUDNN_Op<"batch_norm_inference", [Pure]> {
    let summary = "Batch normalization in inference phase";

    let arguments = (ins
      CUDNN_TensorType:$x,
      CUDNN_TensorType:$scale,
      CUDNN_TensorType:$offset,
      CUDNN_TensorType:$mean,
      CUDNN_TensorType:$variance,
      CUDNN_TensorType:$epsilon
    );

    let results = (outs CUDNN_TensorType:$y);

    let assemblyFormat = [{
      `(` $x `,` $scale `,` $offset `,` $mean `,` $variance `,` $epsilon`)`
         attr-dict `:`
         functional-type(operands, results)
    }];
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
// cudnn.handle operation
//===----------------------------------------------------------------------===//

def CUDNN_HandleOp : CUDNN_Op<"handle", [Pure]> {
  let summary = "Returns a cuDNN handle for the given device";

  let arguments = (ins
    HAL_Device:$device
  );

  let results = (outs
    CUDNN_HandleType:$result
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
// cudnn.call operation
//===----------------------------------------------------------------------===//

def CUDNN_CallOp : CUDNN_Op<"call", [
    DeclareOpInterfaceMethods<CallOpInterface>,
    DeclareOpInterfaceMethods<SymbolUserOpInterface>
  ]> {
  let summary = "cuDNN call operation";

  let description = [{
    Calls a cuDNN graph with the given cuDNN handle and arguments.
  }];

  let arguments = (ins
    CUDNN_GraphRefAttr:$callee,
    CUDNN_HandleType:$handle,
    Variadic<AnyRankedTensor>:$arguments
  );

  let results = (outs
    Variadic<AnyRankedTensor>:$results
  );

  let assemblyFormat = [{
    `handle` `(` $handle `)` $callee    `(` $arguments `)` attr-dict `:`
    functional-type($arguments, $results)
  }];
}

#endif // CUDNN_OPS
