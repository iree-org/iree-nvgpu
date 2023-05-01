module @example {

util.global @handle : !cudnn.handle

util.initializer {
  %device = hal.ex.shared_device : !hal.device
  %handle = cudnn.handle(%device) : !cudnn.handle
  util.global.store %handle, @handle : !cudnn.handle
  util.initializer.return
}

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

func.func @main(
  %x: tensor<8x4x4x32xf32>,
  %w: tensor<32x1x1x32xf32>,
  %b: tensor<8x4x4x32xf32>,
  %c: tensor<1x1x1x32xf32>
) -> tensor<8x4x4x32xf32> {

  %handle = util.global.load @handle : !cudnn.handle

  %0 = cudnn.call handle(%handle) @conv2d(%x, %w, %b, %c)
       : (tensor<8x4x4x32xf32>, tensor<32x1x1x32xf32>,
          tensor<8x4x4x32xf32>, tensor<1x1x1x32xf32>)
       -> tensor<8x4x4x32xf32>

  %1 = mhlo.sqrt %0 : tensor<8x4x4x32xf32>

  return %0 : tensor<8x4x4x32xf32>
}

}
