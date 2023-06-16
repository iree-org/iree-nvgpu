# Copyright 2023 The OpenXLA Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import logging
import unittest

import iree.compiler
import iree.runtime

import numpy as np


def create_simple_mul_module(instance):
  binary = iree.compiler.tools.compile_str("""
    module @conv2d {
      util.global @handle : !cudnn.handle

      util.initializer {
        %device = hal.ex.shared_device : !hal.device
        %handle = cudnn.handle(%device) : !cudnn.handle
        util.global.store %handle, @handle : !cudnn.handle
        util.initializer.return
      }

      cudnn.graph @graph(%x: !cudnn.tensor<8x32x4x4xf32, NHWC>,
                         %w: !cudnn.tensor<32x32x3x3xf32, NHWC>
                        ) -> !cudnn.tensor<8x32x4x4xf32, NHWC> {
        %0 = cudnn.convolution(%x, %w) alpha=1.0 beta=0.0
               spatial_dim_count=2
               spatial_stride=[1,1]
               pre_padding=[1,1]
               post_padding=[1,1]
               dilation=[1,1]
            : (!cudnn.tensor<8x32x4x4xf32, NHWC>,
               !cudnn.tensor<32x32x3x3xf32, NHWC>)
            -> !cudnn.tensor<8x32x4x4xf32, NHWC>
        cudnn.return %0: !cudnn.tensor<8x32x4x4xf32, NHWC>
      }

      func.func @main(%x: tensor<8x4x4x32xf32>,
                      %w: tensor<32x3x3x32xf32>) -> tensor<8x4x4x32xf32> {
        %handle = util.global.load @handle : !cudnn.handle
        %0 = cudnn.call handle(%handle) @graph(%x, %w)
               : (tensor<8x4x4x32xf32>, tensor<32x3x3x32xf32>)
               -> tensor<8x4x4x32xf32>
        return %0 : tensor<8x4x4x32xf32>
      }
    }""",
                                     target_backends=["cuda"],
                                     extra_args=["--iree-plugin=openxla-cudnn"])
  m = iree.runtime.VmModule.from_flatbuffer(instance, binary)
  return m


class Conv2DTest(unittest.TestCase):

  def test_conv2d(self):
    config = iree.runtime.Config("cuda")
    ctx = iree.runtime.SystemContext(config=config)
    self.assertTrue(ctx.is_dynamic)

    ctx.add_module_dependency("cudnn")
    ctx.add_vm_module(create_simple_mul_module(ctx.instance))

    self.assertEqual(ctx.modules.conv2d.name, "conv2d")
    f = ctx.modules.conv2d["main"]

    np.random.seed(0)

    x = np.random.randn(8, 4, 4, 32).astype(np.float32)
    w = np.random.randn(32, 3, 3, 32).astype(np.float32)

    x[np.abs(x) < 1.5] = 1.0
    w[np.abs(w) < 1.5] = 1.0

    result = f(x, w)

    # Compute expected result with numpy
    x_padded = np.pad(x, ((0, 0), (1, 1), (1, 1), (0, 0)))
    expected_result = np.zeros((8, 4, 4, 32))

    for n in range(8):
      for i in range(4):
        for j in range(4):
          for k in range(32):
            expected_result[n, i, j, k] = np.dot(
                x_padded[n, i:i + 3, j:j + 3, :].flatten(),
                w[k, :, :, :].flatten())

    np.testing.assert_allclose(expected_result, result, rtol=1e-3, atol=1e-2)


if __name__ == "__main__":
  logging.basicConfig(level=logging.DEBUG)
  unittest.main()
