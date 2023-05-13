# Copyright 2023 The OpenXLA Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set(TRITON_ROOT "${OPENXLA_NVGPU_ROOT_DIR}/third_party/triton/")
set(TRITON_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}")

set(MLIR_INCLUDE_DIRS
    ${IREE_SOURCE_DIR}/third_party/llvm-project/mlir/include
    ${TRITON_ROOT}/include
)

iree_tablegen_library(
  NAME
    TritonDialectIncGen
  TD_FILE
    "${TRITON_ROOT}/include/triton/Dialect/Triton/IR/TritonDialect.td"
  OUTS
    --gen-dialect-decls include/triton/Dialect/Triton/IR/Dialect.h.inc
    --gen-dialect-defs include/triton/Dialect/Triton/IR/Dialect.cpp.inc
)

iree_tablegen_library(
  NAME
    TritonOpsIncGen
  TD_FILE
    "${TRITON_ROOT}/include/triton/Dialect/Triton/IR/TritonOps.td"
  OUTS
    --gen-enum-decls include/triton/Dialect/Triton/IR/OpsEnums.h.inc
    --gen-enum-defs include/triton/Dialect/Triton/IR/OpsEnums.cpp.inc
    --gen-op-decls include/triton/Dialect/Triton/IR/Ops.h.inc
    --gen-op-defs include/triton/Dialect/Triton/IR/Ops.cpp.inc
    --gen-typedef-decls include/triton/Dialect/Triton/IR/Types.h.inc
    --gen-typedef-defs include/triton/Dialect/Triton/IR/Types.cpp.inc
)

iree_tablegen_library(
  NAME
    TritonInterfacesIncGen
  TD_FILE
    "${TRITON_ROOT}/include/triton/Dialect/Triton/IR/TritonInterfaces.td"
  OUTS
    --gen-attr-interface-decls include/triton/Dialect/Triton/IR/AttrInterfaces.h.inc
    --gen-attr-interface-defs include/triton/Dialect/Triton/IR/AttrInterfaces.cpp.inc
)

iree_tablegen_library(
  NAME
    TritonGpuDialectIncGen
  TD_FILE
    "${TRITON_ROOT}/include/triton/Dialect/TritonGPU/IR/TritonGPUDialect.td"
  OUTS
    --gen-dialect-decls include/triton/Dialect/TritonGPU/IR/Dialect.h.inc
    --gen-dialect-defs include/triton/Dialect/TritonGPU/IR/Dialect.cpp.inc
)

iree_tablegen_library(
  NAME
    TritonGpuOpsIncGen
  TD_FILE
    "${TRITON_ROOT}/include/triton/Dialect/TritonGPU/IR/TritonGPUOps.td"
  OUTS
    --gen-enum-decls include/triton/Dialect/TritonGPU/IR/OpsEnums.h.inc
    --gen-enum-defs include/triton/Dialect/TritonGPU/IR/OpsEnums.cpp.inc
    --gen-op-decls include/triton/Dialect/TritonGPU/IR/Ops.h.inc
    --gen-op-defs include/triton/Dialect/TritonGPU/IR/Ops.cpp.inc
    --gen-typedef-decls include/triton/Dialect/TritonGPU/IR/Types.h.inc
    --gen-typedef-defs include/triton/Dialect/TritonGPU/IR/Types.cpp.inc
)

iree_tablegen_library(
  NAME
    TritonGpuAttrInGen
  TD_FILE
    "${TRITON_ROOT}/include/triton/Dialect/TritonGPU/IR/TritonGPUAttrDefs.td"
  OUTS
    --gen-attrdef-decls include/triton/Dialect/TritonGPU/IR/TritonGPUAttrDefs.h.inc
    --gen-attrdef-defs include/triton/Dialect/TritonGPU/IR/TritonGPUAttrDefs.cpp.inc
)

iree_tablegen_library(
  NAME
    TritonTransformsIncGen
  TD_FILE
    "${TRITON_ROOT}/include/triton/Dialect/Triton/Transforms/Passes.td"
  OUTS
    --gen-pass-decls --name=Triton include/triton/Dialect/Triton/Transforms/Passes.h.inc
)

iree_tablegen_library(
  NAME
    TritonCombineIncGen
  TD_FILE
    "${TRITON_ROOT}/lib/Dialect/Triton/Transforms/Combine.td"
  OUTS
    --gen-rewriters lib/Dialect/Triton/Transforms/TritonCombine.inc
)

iree_tablegen_library(
  NAME
    TritonGpuTransformsIncGen
  TD_FILE
    "${TRITON_ROOT}/include/triton/Dialect/TritonGPU/Transforms/Passes.td"
  OUTS
    --gen-pass-decls --name=TritonGPU include/triton/Dialect/TritonGPU/Transforms/Passes.h.inc
)

iree_tablegen_library(
  NAME
    TritonConversionTritonGpuToLLVMIncGen
  TD_FILE
    "${TRITON_ROOT}/include/triton/Conversion/TritonGPUToLLVM/Passes.td"
  OUTS
    --gen-pass-decls --name=TritonGPUToLLVM include/triton/Conversion/TritonGPUToLLVM/Passes.h.inc
)

iree_tablegen_library(
  NAME
    TritonConversionTritonToTritonGpuIncGen
  TD_FILE
    "${TRITON_ROOT}/include/triton/Conversion/TritonToTritonGPU/Passes.td"
  OUTS
    --gen-pass-decls --name=TritonToTritonGPU include/triton/Conversion/TritonToTritonGPU/Passes.h.inc
)

external_cc_library(
  PACKAGE
    triton
  NAME
    TritonIR
  ROOT
    ${TRITON_ROOT}
  INCLUDES
    "${TRITON_ROOT}/include"
    "${TRITON_BINARY_DIR}/include"
  HDRS
    "include/triton/Dialect/Triton/IR/Dialect.h"
    "include/triton/Dialect/Triton/IR/Interfaces.h"
    "include/triton/Dialect/Triton/IR/Traits.h"
    "include/triton/Dialect/Triton/IR/Types.h"
    "include/triton/Dialect/TritonGPU/IR/Dialect.h"
    "include/triton/Dialect/TritonGPU/IR/Traits.h"
  SRCS
    "lib/Dialect/Triton/IR/Dialect.cpp"
    "lib/Dialect/Triton/IR/Ops.cpp"
    "lib/Dialect/Triton/IR/Traits.cpp"
    "lib/Dialect/Triton/IR/Types.cpp"
    "lib/Dialect/TritonGPU/IR/Dialect.cpp"
    "lib/Dialect/TritonGPU/IR/Traits.cpp"
  DEPS
    build_tools::third_party::triton::TritonDialectIncGen
    build_tools::third_party::triton::TritonOpsIncGen
    build_tools::third_party::triton::TritonInterfacesIncGen
    build_tools::third_party::triton::TritonGpuDialectIncGen
    build_tools::third_party::triton::TritonGpuOpsIncGen
    build_tools::third_party::triton::TritonGpuAttrInGen
    MLIRFuncDialect
    MLIRIR
    MLIRSupport
    MLIRTransformUtils
  COPTS
    # TODO(ezhulenev): Figure out why we are not getting default IREE copts here
    -fno-rtti
    -fno-exceptions
  PUBLIC
)

external_cc_library(
  PACKAGE
    triton
  NAME
    TritonTransforms
  ROOT
    ${TRITON_ROOT}
  INCLUDES
    "${TRITON_ROOT}/include"
    "${TRITON_BINARY_DIR}/include"
    "${TRITON_BINARY_DIR}/lib/Dialect/Triton/Transforms"
  HDRS
    "include/triton/Dialect/Triton/Transforms/Passes.h"
    "include/triton/Dialect/TritonGPU/Transforms/Passes.h"
    "include/triton/Dialect/TritonGPU/Transforms/TritonGPUConversion.h"
  SRCS
    "lib/Dialect/Triton/Transforms/Combine.cpp"
    "lib/Dialect/Triton/Transforms/RewriteTensorPointer.cpp"
    "lib/Dialect/TritonGPU/Transforms/AccelerateMatmul.cpp"
    "lib/Dialect/TritonGPU/Transforms/Coalesce.cpp"
    "lib/Dialect/TritonGPU/Transforms/DecomposeConversions.cpp"
    "lib/Dialect/TritonGPU/Transforms/OptimizeDotOperands.cpp"
    "lib/Dialect/TritonGPU/Transforms/Pipeline.cpp"
    "lib/Dialect/TritonGPU/Transforms/Prefetch.cpp"
    "lib/Dialect/TritonGPU/Transforms/RemoveLayoutConversions.cpp"
    "lib/Dialect/TritonGPU/Transforms/ReorderInstructions.cpp"
    "lib/Dialect/TritonGPU/Transforms/TritonGPUConversion.cpp"
    "lib/Dialect/TritonGPU/Transforms/Utility.cpp"
  DEPS
    build_tools::third_party::triton::TritonTransformsIncGen
    build_tools::third_party::triton::TritonCombineIncGen
    build_tools::third_party::triton::TritonGpuTransformsIncGen
    triton::TritonIR
    triton::TritonAnalysis
    MLIRFuncDialect
    MLIRIR
    MLIRSupport
    MLIRTransformUtils
  COPTS
    # TODO(ezhulenev): Figure out why we are not getting default IREE copts here
    -fno-rtti
    -fno-exceptions
  PUBLIC
)

external_cc_library(
  PACKAGE
    triton
  NAME
    TritonConversion
  ROOT
    ${TRITON_ROOT}
  INCLUDES
    "${TRITON_ROOT}/include"
    "${TRITON_BINARY_DIR}/include"
  HDRS
    "include/triton/Conversion/TritonToTritonGPU/Passes.h"
    "include/triton/Conversion/TritonToTritonGPU/TritonToTritonGPUPass.h"
    "include/triton/Conversion/TritonGPUToLLVM/AsmFormat.h"
    "include/triton/Conversion/TritonGPUToLLVM/GCNAsmFormat.h"
    "include/triton/Conversion/TritonGPUToLLVM/PTXAsmFormat.h"
    "include/triton/Conversion/TritonGPUToLLVM/Passes.h"
    "include/triton/Conversion/TritonGPUToLLVM/TritonGPUToLLVMPass.h"
  SRCS
    "lib/Conversion/TritonToTritonGPU/TritonToTritonGPUPass.cpp"
    "lib/Conversion/TritonGPUToLLVM/TritonGPUToLLVM.cpp"
    "lib/Conversion/TritonGPUToLLVM/GCNAsmFormat.cpp"
    "lib/Conversion/TritonGPUToLLVM/PTXAsmFormat.cpp"
    "lib/Conversion/TritonGPUToLLVM/TritonGPUToLLVMPass.cpp"
    "lib/Conversion/TritonGPUToLLVM/ConvertLayoutOpToLLVM/SharedToDotOperandFMA.cpp"
    "lib/Conversion/TritonGPUToLLVM/ConvertLayoutOpToLLVM/SharedToDotOperandMMAv1.cpp"
    "lib/Conversion/TritonGPUToLLVM/ConvertLayoutOpToLLVM/SharedToDotOperandMMAv2.cpp"
    "lib/Conversion/TritonGPUToLLVM/ConvertLayoutOpToLLVM.cpp"
    "lib/Conversion/TritonGPUToLLVM/DotOpToLLVM/FMA.cpp"
    "lib/Conversion/TritonGPUToLLVM/DotOpToLLVM/MMAv1.cpp"
    "lib/Conversion/TritonGPUToLLVM/DotOpToLLVM/MMAv2.cpp"
    "lib/Conversion/TritonGPUToLLVM/DotOpToLLVM.cpp"
    "lib/Conversion/TritonGPUToLLVM/ElementwiseOpToLLVM.cpp"
    "lib/Conversion/TritonGPUToLLVM/LoadStoreOpToLLVM.cpp"
    "lib/Conversion/TritonGPUToLLVM/TritonGPUToLLVM.cpp"
    "lib/Conversion/TritonGPUToLLVM/TritonGPUToLLVMPass.cpp"
    "lib/Conversion/TritonGPUToLLVM/PTXAsmFormat.cpp"
    "lib/Conversion/TritonGPUToLLVM/ReduceOpToLLVM.cpp"
    "lib/Conversion/TritonGPUToLLVM/Utility.cpp"
    "lib/Conversion/TritonGPUToLLVM/TypeConverter.cpp"
    "lib/Conversion/TritonGPUToLLVM/ViewOpToLLVM.cpp"
  DEPS
    build_tools::third_party::triton::TritonConversionTritonToTritonGpuIncGen
    build_tools::third_party::triton::TritonConversionTritonGpuToLLVMIncGen
    triton::TritonIR
    triton::TritonAnalysis
    MLIRFuncDialect
    MLIRIR
    MLIRSupport
    MLIRTransformUtils
  COPTS
    # TODO(ezhulenev): Figure out why we are not getting default IREE copts here
    -fno-rtti
    -fno-exceptions
  PUBLIC
)

external_cc_library(
  PACKAGE
    triton
  NAME
    TritonAnalysis
  ROOT
    ${TRITON_ROOT}
  INCLUDES
    "${TRITON_ROOT}/include"
    "${TRITON_BINARY_DIR}/include"
  HDRS
    "include/triton/Analysis/Alias.h"
    "include/triton/Analysis/Allocation.h"
    "include/triton/Analysis/AxisInfo.h"
    "include/triton/Analysis/Membar.h"
    "include/triton/Analysis/Utility.h"
  SRCS
    "lib/Analysis/Alias.cpp"
    "lib/Analysis/Allocation.cpp"
    "lib/Analysis/AxisInfo.cpp"
    "lib/Analysis/Membar.cpp"
    "lib/Analysis/Utility.cpp"
  DEPS
    build_tools::third_party::triton::TritonGpuAttrInGen
    triton::TritonIR
    MLIRFuncDialect
    MLIRIR
    MLIRSupport
    MLIRTransformUtils
  COPTS
    # TODO(ezhulenev): Figure out why we are not getting default IREE copts here
    -fno-rtti
    -fno-exceptions
  PUBLIC
)

external_cc_library(
  PACKAGE
    triton
  NAME
    TritonLLVMIR
  ROOT
    ${TRITON_ROOT}
  INCLUDES
    "${TRITON_ROOT}/include"
    "${TRITON_BINARY_DIR}/include"
  HDRS
    "include/triton/Target/LLVMIR/LLVMIRTranslation.h"
  SRCS
    "lib/Target/LLVMIR/LLVMIRTranslation.cpp"
  DEPS
    LLVMIRReader
    MLIRLLVMDialect
    MLIRROCDLToLLVMIRTranslation
    MLIRIR
    MLIRSupport
    MLIRTransformUtils
  COPTS
    # TODO(ezhulenev): Figure out why we are not getting default IREE copts here
    -fno-rtti
    -fno-exceptions
  PUBLIC
)

external_cc_library(
  PACKAGE
    triton
  NAME
    TritonPTX
  ROOT
    ${TRITON_ROOT}
  INCLUDES
    "${TRITON_ROOT}/include"
    "${TRITON_BINARY_DIR}/include"
  HDRS
    "include/triton/Target/PTX/PTXTranslation.h"
  SRCS
    "lib/Target/PTX/PTXTranslation.cpp"
  DEPS
    LLVMSupport
    MLIRLLVMDialect
    MLIRIR
    MLIRSupport
    MLIRTransformUtils
  COPTS
    # TODO(ezhulenev): Figure out why we are not getting default IREE copts here
    -fno-rtti
    -fno-exceptions
  PUBLIC
)
