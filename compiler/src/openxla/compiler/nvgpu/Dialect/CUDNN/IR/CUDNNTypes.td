//===- CUDNNTypes.td - CUDNN dialect types ------------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CUDNN_TYPES
#define CUDNN_TYPES

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"
include "openxla/compiler/nvgpu/Dialect/CUDNN/IR/CUDNNDialect.td"

//===----------------------------------------------------------------------===//
// CUDNN type definitions
//===----------------------------------------------------------------------===//

class CUDNN_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<CUDNN_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def CUDNN_ExecutionPlanType : CUDNN_Type<"ExecutionPlan", "execution_plan"> {
    let summary = "CUDNN Execution Plan";
}

def CUDNN_OperationGraphType : CUDNN_Type<"OperationGraph", "operation_graph"> {
    let summary = "CUDNN Operation Graph";
    let description = [{
      Handle to graph of operations that will be performed.
    }];
}

def CUDNN_HandleType : CUDNN_Type<"Handle", "handle"> {
    let summary = "CUDNN handle";
}

def CUDNN_StatusType : CUDNN_Type<"Status", "status"> {
    let summary = "CUDNN Status type";
    let description = "Return code, corresponding to cudnnStatus_t";
}

def CUDNN_TensorDescType : CUDNN_Type<"TensorDesc", "tensor_desc"> {
    let summary = "CUDNN Tensor descriptor";
    let description = [{
      Description of the memory layout, element type, alignment and stride of
      the input/results.
    }];
    let parameters = (ins
      ArrayRefParameter<"int64_t">:$shape,
      "Type":$element_type,
      "int":$alignment,
       ArrayRefParameter<"int64_t">:$stride
    );
    // TODO: This is currently order dependent, remove the need for named parts.
    let assemblyFormat = [{
      `<` custom<DimensionList>($shape, $element_type) `,` `alignment` `=` $alignment `,`
      `stride` `=` `[` $stride `]` `>`
    }];
}

def CUDNN_ExecutableType : CUDNN_Type<"Executable", "executable"> {
    let summary = "cuDNN Executable";
}

//===----------------------------------------------------------------------===//
// cuDNN data type corresponding to cudnnDataType_t
//===----------------------------------------------------------------------===//

def CUDNN_DATATYPE_FLOAT    : I32EnumAttrCase<"FLOAT", 0>;
def CUDNN_DATATYPE_DOUBLE   : I32EnumAttrCase<"DOUBLE", 1>;
def CUDNN_DATATYPE_HALF     : I32EnumAttrCase<"HALF", 2>;
def CUDNN_DATATYPE_INT8     : I32EnumAttrCase<"INT8", 3>;
def CUDNN_DATATYPE_INT32    : I32EnumAttrCase<"INT32", 4>;
def CUDNN_DATATYPE_INT8x4   : I32EnumAttrCase<"INT8x4", 5>;
def CUDNN_DATATYPE_UINT8    : I32EnumAttrCase<"UINT8", 6>;
def CUDNN_DATATYPE_UINT8x4  : I32EnumAttrCase<"UINT8x4", 7>;
def CUDNN_DATATYPE_INT8x32  : I32EnumAttrCase<"INT8x32", 8>;
def CUDNN_DATATYPE_BFLOAT16 : I32EnumAttrCase<"BFLOAT16", 9>;
def CUDNN_DATATYPE_INT64    : I32EnumAttrCase<"INT64", 10>;
def CUDNN_DATATYPE_BOOLEAN  : I32EnumAttrCase<"BOOLEAN", 11>;

def CUDNN_DATATYPE : I32EnumAttr<"DataType",
    "cuDNN data type",
    [
      CUDNN_DATATYPE_FLOAT,
      CUDNN_DATATYPE_DOUBLE,
      CUDNN_DATATYPE_HALF,
      CUDNN_DATATYPE_INT8,
      CUDNN_DATATYPE_INT32,
      CUDNN_DATATYPE_INT8x4,
      CUDNN_DATATYPE_UINT8,
      CUDNN_DATATYPE_UINT8x4,
      CUDNN_DATATYPE_INT8x32,
      CUDNN_DATATYPE_BFLOAT16,
      CUDNN_DATATYPE_INT64,
      CUDNN_DATATYPE_BOOLEAN
    ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::openxla::compiler::nvgpu::cudnn";
}

//===----------------------------------------------------------------------===//
// cuDNN data layout formats
//===----------------------------------------------------------------------===//

// cuDNN supports a generic way of specifying tensor layouts with strides,
// however there are a few widely used data layouts supported by cuDNN
// operations, that we define as an enum attribute to avoid always passing
// around affine maps.

// 4-D Tensor layouts.
//
// In cuDNN 4-D tensors always have their logical dimensions as NCHW (batch,
// channels aka feature maps, height, width). Tensor layout defines how these
// logical dimensions layed out in memory.
//
// Example:
//   `!cudnn.tensor<1x64x32x32xf32, NHWC>` is equivalent to `memref<1x32x32x64>`
//   physical memory layout
//
// NCHW tensor layout is a simple row-major layout.
def CUDNN_LAYOUT_NCHW : I32EnumAttrCase<"NCHW", 1>;
def CUDNN_LAYOUT_NHWC : I32EnumAttrCase<"NHWC", 2>;

def CUDNN_Layout : I32EnumAttr<"Layout",
    "cuDNN data layout format",
    [
      CUDNN_LAYOUT_NCHW,
      CUDNN_LAYOUT_NHWC
    ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::openxla::compiler::nvgpu::cudnn";
}

def CUDNN_LayoutAttr : EnumAttr<CUDNN_Dialect, CUDNN_Layout, "layout">;

//===----------------------------------------------------------------------===//
// !cudnn.tensor type
//===----------------------------------------------------------------------===//

def CUDNN_TensorType : CUDNN_Type<"Tensor", "tensor"> {
    let summary = "cuDNN Tensor";
    let description = [{
      CuDNN tensor type describing memory shape, data type and layout. This type
      corresponds to the cuDNN tensor descriptor, however at run time cuDNN
      tensors have additional parameters that are not know or important at
      compile time (e.g. `virtual` flag and memory alignment).

      Shape and layout can be omitted from the type when lowering to the runtime
      function calls (just a `!cudnn.tensor`). At run time, shape, type and
      layout become a property of reference counted runtime values.

      See cuDNN documentation:
      https://docs.nvidia.com/deeplearning/cudnn/developer-guide/index.html#tensors-layouts
    }];

    let parameters = (ins
      ArrayRefParameter<"int64_t">:$shape,
      "Type":$elementType,
      "std::optional<Layout>":$layout,
      "mlir::AffineMap":$strides
    );

    let builders = [
      TypeBuilder<(ins), [{
        return $_get($_ctxt, ArrayRef<int64_t>(), Type(),
                     std::nullopt, AffineMap());
      }]>,
      TypeBuilderWithInferredContext<(ins "llvm::ArrayRef<int64_t>":$shape,
                                          "Type":$elementType), [{
        return $_get(elementType.getContext(), shape, elementType,
                     std::nullopt, AffineMap());
      }]>,
      TypeBuilderWithInferredContext<(ins "llvm::ArrayRef<int64_t>":$shape,
                                          "Type":$elementType,
                                          "Layout":$layout), [{
        return $_get(elementType.getContext(), shape, elementType,
                     layout, AffineMap());
      }]>,
      TypeBuilderWithInferredContext<(ins "llvm::ArrayRef<int64_t>":$shape,
                                          "Type":$elementType,
                                          "mlir::AffineMap":$strides), [{
        return $_get(elementType.getContext(), shape, elementType,
                     std::nullopt, strides);
      }]>
    ];

    let extraClassDeclaration = [{
      bool isOpaque();
    }];

    let hasCustomAssemblyFormat = 1;
    let skipDefaultBuilders = 1;
    let genVerifyDecl = 1;
}

#endif // CUDNN_TYPES
