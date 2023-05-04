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
    "conv2d_add_bias_cudnn.mlir"
    "conv2d_add_bias_mhlo.mlir"
  TOOLS
    FileCheck
    iree-compile
  LABELS
    "driver=cuda"
)
