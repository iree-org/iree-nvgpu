# Copyright 2023 The OpenXLA Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

if(NOT IREE_HAL_DRIVER_CUDA OR NOT IREE_TARGET_BACKEND_CUDA)
  return()
endif()

iree_lit_test_suite(
  NAME
    lit
  SRCS
    "triton_add_bytecode.mlir"
    "triton_add_inplace_bytecode.mlir"
    "triton_add_inplace.mlir"
    "triton_add.mlir"
    "triton_custom_call_add.mlir"
    "triton_custom_call_add_inplace.mlir"
  TOOLS
    FileCheck
    iree-compile
    iree-run-module
  LABELS
    "driver=cuda"
)
