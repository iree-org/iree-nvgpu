# Copyright 2023 The OpenXLA Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

iree_add_all_subdirs()

iree_cc_library(
  NAME
    HloToTriton
  HDRS
    "ConvertHloToTriton.h"
  SRCS
    "ConvertHloToTriton.cpp"
  DEPS
    LLVMSupport
    MLIRIR
    MLIRMhloUtils
    MLIRPass
    MLIRTransforms
    MhloDialect
    openxla::compiler::nvgpu::Dialect::TritonFlow::IR
    tensorflow::external_mhlo_includes
    # TODO(ezhulenev): Triton dependencies are not part of bazel build.
    triton::TritonIR
  PUBLIC
)

iree_cc_library(
  NAME
    TritonToFlowDispatch
  HDRS
    "ConvertTritonToFlowDispatch.h"
  SRCS
    "ConvertTritonToFlowDispatch.cpp"
  DEPS
    LLVMSupport
    MLIRIR
    MLIRPass
    MLIRTransforms
    iree::compiler::Dialect::Flow::IR
    iree::compiler::Dialect::HAL::IR
    iree::compiler::Utils
    openxla::compiler::nvgpu::Dialect::TritonFlow::IR
    # TODO(ezhulenev): Triton dependencies are not part of bazel build.
    triton::TritonIR
    triton::TritonTransforms
    triton::TritonConversion
    triton::TritonLLVMIR
    triton::TritonPTX
  PUBLIC
)
