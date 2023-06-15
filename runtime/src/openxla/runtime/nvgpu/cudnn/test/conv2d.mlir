// RUN: iree-compile --iree-plugin=openxla-cudnn --iree-input-type=mhlo        \
// RUN:              --iree-hal-target-backends=cuda %s                        \
// RUN: | iree-run-module --module=- --device=cuda --function=main             \
// RUN: | FileCheck %s

util.global @handle : !cudnn.handle

util.initializer {
  %device = hal.ex.shared_device : !hal.device
  %handle = cudnn.handle(%device) : !cudnn.handle
  util.global.store %handle, @handle : !cudnn.handle
  util.initializer.return
}

cudnn.graph @conv2d(%x: !cudnn.tensor<8x32x4x4xf64, NHWC>,
                    %w: !cudnn.tensor<32x32x1x1xf64, KHWC>)
                    -> !cudnn.tensor<8x32x4x4xf64, NHWC> {
  %0 = cudnn.convolution(%x, %w) alpha=1.0 beta=0.0
      spatial_dim_count=2
      spatial_stride=[1,1]
      pre_padding=[0,0]
      post_padding=[0,0]
      dilation=[1,1]
      : (!cudnn.tensor<8x32x4x4xf64, NHWC>, !cudnn.tensor<32x32x1x1xf64, KHWC>)
      -> !cudnn.tensor<8x32x4x4xf64, NHWC>
  cudnn.return %0: !cudnn.tensor<8x32x4x4xf64, NHWC>
}

util.global @x : tensor<8x4x4x32xf64> = dense<1.0> : tensor<8x4x4x32xf64>
util.global @w : tensor<32x1x1x32xf64> = dense<1.0> : tensor<32x1x1x32xf64>

// CHECK: EXEC @main
// CHECK: result[0]: hal.buffer_view
// CHECK: 8x4x4x32xf64
// CHECK: [32 32 32 32 32
func.func @main() -> tensor<8x4x4x32xf64> {
  %x = util.global.load @x : tensor<8x4x4x32xf64>
  %w = util.global.load @w : tensor<32x1x1x32xf64>
  %handle = util.global.load @handle : !cudnn.handle

  %0 = cudnn.call handle(%handle) @conv2d(%x, %w)
       : (tensor<8x4x4x32xf64>, tensor<32x1x1x32xf64>) -> tensor<8x4x4x32xf64>

  return %0 : tensor<8x4x4x32xf64>
}
