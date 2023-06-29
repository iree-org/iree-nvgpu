// RUN: iree-compile --iree-plugin=openxla-cudnn --iree-input-type=stablehlo   \
// RUN:              --iree-hal-target-backends=cuda %s                        \
// RUN: | iree-run-module --module=- --device=cuda --function=run.conv2d_3x3   \
// RUN: | FileCheck %s

util.global @handle : !cudnn.handle

util.initializer {
  %device = hal.ex.shared_device : !hal.device
  %handle = cudnn.handle(%device) : !cudnn.handle
  util.global.store %handle, @handle : !cudnn.handle
  util.initializer.return
}

util.global @x : tensor<8x4x4x32xf32> = dense<1.0> : tensor<8x4x4x32xf32>
util.global @b : tensor<8x4x4x32xf32> = dense<2.0> : tensor<8x4x4x32xf32>
util.global @c : tensor<1x1x1x32xf32> = dense<0.5> : tensor<1x1x1x32xf32>

//===-----------------------------------------------------------------------===/
// 3x3 convolution
//===-----------------------------------------------------------------------===/

util.global @w_3x3 : tensor<32x3x3x32xf32> = dense<1.0> : tensor<32x3x3x32xf32>

// CHECK: result[0]: hal.buffer_view
// CHECK: 130 130 130 130 130 130 130
// CHECK: 194 194 194 194 194 194 194
// CHECK: 290 290 290 290 290 290 290
cudnn.graph @conv2d_3x3(%x: !cudnn.tensor<8x32x4x4xf32, NHWC>,
                        %w: !cudnn.tensor<32x32x3x3xf32, NHWC>,
                        %b: !cudnn.tensor<8x32x4x4xf32, NHWC>,
                        %c: !cudnn.tensor<1x32x1x1xf32, NHWC>)
                         -> !cudnn.tensor<8x32x4x4xf32, NHWC> {
  %0 = cudnn.convolution(%x, %w) alpha=1.0 beta=0.0
      spatial_dim_count=2
      spatial_stride=[1,1]
      pre_padding=[1,1]
      post_padding=[1,1]
      dilation=[1,1]
      : (!cudnn.tensor<8x32x4x4xf32, NHWC>, !cudnn.tensor<32x32x3x3xf32, NHWC>)
      -> !cudnn.tensor<8x32x4x4xf32, NHWC>

  %1 = cudnn.add(%0, %b) alpha=1.0 alpha2=0.75
      : (!cudnn.tensor<8x32x4x4xf32, NHWC>, !cudnn.tensor<8x32x4x4xf32, NHWC>)
      -> !cudnn.tensor<8x32x4x4xf32, NHWC>

  %2 = cudnn.bias(%1, %c)
      : (!cudnn.tensor<8x32x4x4xf32, NHWC>, !cudnn.tensor<1x32x1x1xf32, NHWC>)
      -> !cudnn.tensor<8x32x4x4xf32, NHWC>

  cudnn.return %2: !cudnn.tensor<8x32x4x4xf32, NHWC>
}

func.func @run.conv2d_3x3() -> tensor<8x4x4x32xf32> {
  %x = util.global.load @x : tensor<8x4x4x32xf32>
  %b = util.global.load @b : tensor<8x4x4x32xf32>
  %c = util.global.load @c : tensor<1x1x1x32xf32>

  %w = util.global.load @w_3x3 : tensor<32x3x3x32xf32>

  %handle = util.global.load @handle : !cudnn.handle

  %0 = cudnn.call handle(%handle) @conv2d_3x3(%x, %w, %b, %c)
      : (tensor<8x4x4x32xf32>, tensor<32x3x3x32xf32>, tensor<8x4x4x32xf32>,
      tensor<1x1x1x32xf32>) -> tensor<8x4x4x32xf32>

  return %0 : tensor<8x4x4x32xf32>
}
