module @example {

cudnn.graph @conv2d(%x: !cudnn.tensor<8x32x4x4xf32, NHWC>,
                    %w: !cudnn.tensor<32x32x1x1xf32, NHWC>,
                    %b: !cudnn.tensor<8x32x4x4xf32, NHWC>,
                    %c: !cudnn.tensor<1x32x1x1xf32, NHWC>)
                     -> !cudnn.tensor<8x32x4x4xf32, NHWC> {
  %0 = cudnn.convolution(%x, %w) alpha=1.0 beta=0.0
         spatial_dim_count=2
         spatial_stride=[1,1]
         pre_padding=[0,0]
         post_padding=[0,0]
         dilation=[1,1]
    : (!cudnn.tensor<8x32x4x4xf32, NHWC>, !cudnn.tensor<32x32x1x1xf32, NHWC>)
    -> !cudnn.tensor<8x32x4x4xf32, NHWC>

  %1 = cudnn.add(%0, %b) alpha=1.0 alpha2=0.75
    : (!cudnn.tensor<8x32x4x4xf32, NHWC>, !cudnn.tensor<8x32x4x4xf32, NHWC>)
    -> !cudnn.tensor<8x32x4x4xf32, NHWC>

  %2 = cudnn.bias(%1, %c)
    : (!cudnn.tensor<8x32x4x4xf32, NHWC>, !cudnn.tensor<1x32x1x1xf32, NHWC>)
    -> !cudnn.tensor<8x32x4x4xf32, NHWC>

  cudnn.return %2: !cudnn.tensor<8x32x4x4xf32, NHWC>
}

util.global @x : tensor<8x4x4x32xf32> = dense<1.0> : tensor<8x4x4x32xf32>
util.global @w : tensor<32x1x1x32xf32> = dense<1.0> : tensor<32x1x1x32xf32>
util.global @b : tensor<8x4x4x32xf32> = dense<2.0> : tensor<8x4x4x32xf32>
util.global @c : tensor<1x1x1x32xf32> = dense<0.5> : tensor<1x1x1x32xf32>

func.func @main() -> tensor<8x4x4x32xf32> {
  %x = util.global.load @x : tensor<8x4x4x32xf32>
  %w = util.global.load @w : tensor<32x1x1x32xf32>
  %b = util.global.load @b : tensor<8x4x4x32xf32>
  %c = util.global.load @c : tensor<1x1x1x32xf32>

  %device = hal.ex.shared_device : !hal.device
  %handle = cudnn.handle(%device) : !cudnn.handle

  %0 = cudnn.call handle(%handle) @conv2d(%x, %w, %b, %c)
       : (tensor<8x4x4x32xf32>, tensor<32x1x1x32xf32>,
          tensor<8x4x4x32xf32>, tensor<1x1x1x32xf32>)
       -> tensor<8x4x4x32xf32>

  return %0 : tensor<8x4x4x32xf32>
}

}
