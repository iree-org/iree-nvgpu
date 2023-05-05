# Copyright 2023 The OpenXLA Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

iree_add_all_subdirs()

iree_tablegen_library(
  NAME
    PassesIncGen
  TD_FILE
    "Passes.td"
  OUTS
    --gen-pass-decls Passes.h.inc
)

iree_cc_library(
  NAME
    PassHeaders
    HDRS
    "PassDetail.h"
    "Passes.h"
    "Passes.h.inc"
  DEPS
    ::PassesIncGen
    MLIRPass
    MLIRTransforms
  PUBLIC
)

iree_cc_library(
  NAME
    Transforms
  HDRS
    "ConvertCUDNNToRuntime.h"
    "Passes.h"
  SRCS
    "ConvertCUDNNToRuntime.cpp"
    "ExpandCUDNNOperations.cpp"
  DEPS
    ::PassHeaders
    LLVMSupport
    MLIRFuncDialect
    MLIRIR
    MLIRPass
    MLIRTransforms
    openxla::compiler::nvgpu::Dialect::CUDNN::IR
    iree::compiler::Dialect::HAL::IR
  PUBLIC
)
