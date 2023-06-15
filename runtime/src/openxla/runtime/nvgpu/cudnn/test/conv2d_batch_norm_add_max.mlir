// RUN: iree-compile --iree-plugin=openxla-cudnn --iree-input-type=stablehlo   \
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

cudnn.graph @conv2d(%x: !cudnn.tensor<8x32x4x4xf32, NHWC>,
                    %w: !cudnn.tensor<32x32x1x1xf32, NHWC>,
                    %scale: !cudnn.tensor<1x32x1x1xf32, NHWC>,
                    %offset: !cudnn.tensor<1x32x1x1xf32, NHWC>,
                    %mean: !cudnn.tensor<1x32x1x1xf32, NHWC>,
                    %var: !cudnn.tensor<1x32x1x1xf32, NHWC>,
                    %epsilon: !cudnn.tensor<1x1x1x1xf32, NHWC>,
                    %max: !cudnn.tensor<8x32x4x4xf32, NHWC>
  ) -> !cudnn.tensor<8x32x4x4xf32, NHWC> {
  %0 = cudnn.convolution(%x, %w) alpha=1.0 beta=0.0
         spatial_dim_count=2
         spatial_stride=[1,1]
         pre_padding=[0,0]
         post_padding=[0,0]
         dilation=[1,1]
    : (!cudnn.tensor<8x32x4x4xf32, NHWC>, !cudnn.tensor<32x32x1x1xf32, NHWC>)
    -> !cudnn.tensor<8x32x4x4xf32, NHWC>

  %1 = cudnn.batch_norm_inference(%0, %scale, %offset, %mean, %var, %epsilon)
      : (!cudnn.tensor<8x32x4x4xf32, NHWC>, !cudnn.tensor<1x32x1x1xf32, NHWC>,
         !cudnn.tensor<1x32x1x1xf32, NHWC>, !cudnn.tensor<1x32x1x1xf32, NHWC>,
         !cudnn.tensor<1x32x1x1xf32, NHWC>, !cudnn.tensor<1x1x1x1xf32, NHWC>)
      -> !cudnn.tensor<8x32x4x4xf32, NHWC>

  %2 = cudnn.add(%1, %x)
     : (!cudnn.tensor<8x32x4x4xf32, NHWC>, !cudnn.tensor<8x32x4x4xf32, NHWC>)
     -> !cudnn.tensor<8x32x4x4xf32, NHWC>

  %3 = cudnn.max(%2, %max)
     : (!cudnn.tensor<8x32x4x4xf32, NHWC>, !cudnn.tensor<8x32x4x4xf32, NHWC>)
     -> !cudnn.tensor<8x32x4x4xf32, NHWC>

  cudnn.return %3: !cudnn.tensor<8x32x4x4xf32, NHWC>
}

util.global @x : tensor<8x4x4x32xf32> = dense<1.0> : tensor<8x4x4x32xf32>
util.global @w : tensor<32x1x1x32xf32> = dense<1.0> : tensor<32x1x1x32xf32>
util.global @scale : tensor<1x1x1x32xf32> = dense<0.5> : tensor<1x1x1x32xf32>
util.global @offset : tensor<1x1x1x32xf32> = dense<1.0> : tensor<1x1x1x32xf32>
util.global @mean : tensor<1x1x1x32xf32> = dense<0.25> : tensor<1x1x1x32xf32>
util.global @variance : tensor<1x1x1x32xf32> = dense<0.2> : tensor<1x1x1x32xf32>
util.global @epsilon : tensor<1x1x1x1xf32> = dense<0.0001> : tensor<1x1x1x1xf32>
util.global @max : tensor<8x4x4x32xf32> = dense<38.0> : tensor<8x4x4x32xf32>

// CHECK: result[0]: hal.buffer_view
// CHECK: 38 38 38 38 38 38 38 38 38
func.func @main() -> tensor<8x4x4x32xf32> {
  %x = util.global.load @x : tensor<8x4x4x32xf32>
  %w = util.global.load @w : tensor<32x1x1x32xf32>
  %scale = util.global.load @scale : tensor<1x1x1x32xf32>
  %offset = util.global.load @offset : tensor<1x1x1x32xf32>
  %mean = util.global.load @mean : tensor<1x1x1x32xf32>
  %var = util.global.load @variance : tensor<1x1x1x32xf32>
  %epsilon = util.global.load @epsilon : tensor<1x1x1x1xf32>
  %max = util.global.load @max : tensor<8x4x4x32xf32>

  %handle = util.global.load @handle : !cudnn.handle

  %0 = cudnn.call handle(%handle)
                  @conv2d(%x, %w, %scale, %offset, %mean, %var, %epsilon, %max)
       : (tensor<8x4x4x32xf32>, tensor<32x1x1x32xf32>, tensor<1x1x1x32xf32>,
          tensor<1x1x1x32xf32>, tensor<1x1x1x32xf32>, tensor<1x1x1x32xf32>,
          tensor<1x1x1x1xf32>, tensor<8x4x4x32xf32>)
       -> tensor<8x4x4x32xf32>

  return %0 : tensor<8x4x4x32xf32>
}
