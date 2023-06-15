// RUN: iree-compile %s --iree-plugin=openxla-cudnn                            \
// RUN:     --iree-input-type=stablehlo --compile-to=vm                        \
// RUN:     --iree-hal-target-backends=cuda                                    \
// RUN: | FileCheck %s

util.global @handle : !cudnn.handle

util.initializer {
  %device = hal.ex.shared_device : !hal.device
  %handle = cudnn.handle(%device) : !cudnn.handle
  util.global.store %handle, @handle : !cudnn.handle
  util.initializer.return
}

//===-----------------------------------------------------------------------===/
// 1x1 convolution
//===-----------------------------------------------------------------------===/

// CHECK: vm.func private @cudnn.conv2d_1x1.builder
cudnn.graph @cudnn.conv2d_1x1(%x: !cudnn.tensor<8x32x256x256xf32, NHWC>,
                              %w: !cudnn.tensor<32x32x1x1xf32, NHWC>,
                              %b: !cudnn.tensor<8x32x256x256xf32, NHWC>,
                              %c: !cudnn.tensor<1x32x1x1xf32, NHWC>)
                               -> !cudnn.tensor<8x32x256x256xf32, NHWC> {
  %0 = cudnn.convolution(%x, %w) alpha=1.0 beta=0.0
         spatial_dim_count=2
         spatial_stride=[1,1]
         pre_padding=[0,0]
         post_padding=[0,0]
         dilation=[1,1]
    : (!cudnn.tensor<8x32x256x256xf32, NHWC>,
       !cudnn.tensor<32x32x1x1xf32, NHWC>)
    -> !cudnn.tensor<8x32x256x256xf32, NHWC>

  %1 = cudnn.add(%0, %b) alpha=1.0 alpha2=1.0
    : (!cudnn.tensor<8x32x256x256xf32, NHWC>,
       !cudnn.tensor<8x32x256x256xf32, NHWC>)
    -> !cudnn.tensor<8x32x256x256xf32, NHWC>

  %2 = cudnn.bias(%1, %c)
    : (!cudnn.tensor<8x32x256x256xf32, NHWC>,
       !cudnn.tensor<1x32x1x1xf32, NHWC>)
    -> !cudnn.tensor<8x32x256x256xf32, NHWC>

  cudnn.return %2: !cudnn.tensor<8x32x256x256xf32, NHWC>
}

// CHECK: vm.func private @conv2d_1x1
func.func @conv2d_1x1(
  %x: tensor<8x256x256x32xf32>,
  %w: tensor<32x1x1x32xf32>,
  %b: tensor<8x256x256x32xf32>,
  %c: tensor<1x1x1x32xf32>
) -> tensor<8x256x256x32xf32> {

  %handle = util.global.load @handle : !cudnn.handle

  %0 = cudnn.call handle(%handle) @cudnn.conv2d_1x1(%x, %w, %b, %c)
       : (tensor<8x256x256x32xf32>, tensor<32x1x1x32xf32>,
          tensor<8x256x256x32xf32>, tensor<1x1x1x32xf32>)
       -> tensor<8x256x256x32xf32>

  %1 = mhlo.sqrt %0 : tensor<8x256x256x32xf32>

  return %0 : tensor<8x256x256x32xf32>
}

//===-----------------------------------------------------------------------===/
// 3x3 convolution
//===-----------------------------------------------------------------------===/

// CHECK: vm.func private @cudnn.conv2d_3x3.builder
cudnn.graph @cudnn.conv2d_3x3(%x: !cudnn.tensor<8x32x256x256xf32, NHWC>,
                              %w: !cudnn.tensor<32x32x3x3xf32, NHWC>,
                              %b: !cudnn.tensor<8x32x256x256xf32, NHWC>,
                              %c: !cudnn.tensor<1x32x1x1xf32, NHWC>)
                               -> !cudnn.tensor<8x32x256x256xf32, NHWC> {
  %0 = cudnn.convolution(%x, %w) alpha=1.0 beta=0.0
         spatial_dim_count=2
         spatial_stride=[1,1]
         pre_padding=[1,1]
         post_padding=[1,1]
         dilation=[1,1]
    : (!cudnn.tensor<8x32x256x256xf32, NHWC>,
       !cudnn.tensor<32x32x3x3xf32, NHWC>)
    -> !cudnn.tensor<8x32x256x256xf32, NHWC>

  %1 = cudnn.add(%0, %b) alpha=1.0 alpha2=1.0
    : (!cudnn.tensor<8x32x256x256xf32, NHWC>,
       !cudnn.tensor<8x32x256x256xf32, NHWC>)
    -> !cudnn.tensor<8x32x256x256xf32, NHWC>

  %2 = cudnn.bias(%1, %c)
    : (!cudnn.tensor<8x32x256x256xf32, NHWC>,
       !cudnn.tensor<1x32x1x1xf32, NHWC>)
    -> !cudnn.tensor<8x32x256x256xf32, NHWC>

  cudnn.return %2: !cudnn.tensor<8x32x256x256xf32, NHWC>
}

// CHECK: vm.func private @conv2d_3x3
func.func @conv2d_3x3(
  %x: tensor<8x256x256x32xf32>,
  %w: tensor<32x3x3x32xf32>,
  %b: tensor<8x256x256x32xf32>,
  %c: tensor<1x1x1x32xf32>
) -> tensor<8x256x256x32xf32> {

  %handle = util.global.load @handle : !cudnn.handle

  %0 = cudnn.call handle(%handle) @cudnn.conv2d_3x3(%x, %w, %b, %c)
       : (tensor<8x256x256x32xf32>, tensor<32x3x3x32xf32>,
          tensor<8x256x256x32xf32>, tensor<1x1x1x32xf32>)
       -> tensor<8x256x256x32xf32>

  %1 = mhlo.sqrt %0 : tensor<8x256x256x32xf32>

  return %0 : tensor<8x256x256x32xf32>
}
