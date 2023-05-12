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
    "conv2d_1x1_add_bias.mlir"
    "conv2d_3x3_add_bias.mlir"
    "conv2d_batch_norm_add_max.mlir"
    "conv2d_batch_norm_expanded.mlir"
    "conv2d_batch_norm.mlir"
    "conv2d.mlir"
    "e2e_conv2d.mlir"
    "e2e_resnet50_fake_weights.mlir"
  TOOLS
    FileCheck
    iree-compile
    iree-run-module
  LABELS
    "driver=cuda"
)
