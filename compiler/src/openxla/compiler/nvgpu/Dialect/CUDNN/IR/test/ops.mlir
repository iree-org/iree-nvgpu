// RUN: iree-opt --iree-plugin=openxla_nvgpu --split-input-file %s \
// RUN:   | iree-opt --iree-plugin=openxla_nvgpu --split-input-file \
// RUN:   | FileCheck %s

cudnn.graph @graph(%arg0: !cudnn.tensor<1x4x8xf32>)
                     -> !cudnn.tensor<1x4x8xf32> {
  cudnn.return %arg0: !cudnn.tensor<1x4x8xf32>
}

func.func @main(%arg0: tensor<1x4x8xf32>) -> tensor<1x4x8xf32> {
  %0 = cudnn.call @graph(%arg0) : (tensor<1x4x8xf32>) -> tensor<1x4x8xf32>
  return %0 : tensor<1x4x8xf32>
}

// -----

cudnn.graph @graph(%arg0: !cudnn.tensor<1x32x4x4xf32, NHWC>)
                       -> !cudnn.tensor<1x32x4x4xf32, NHWC> {
  cudnn.return %arg0: !cudnn.tensor<1x32x4x4xf32, NHWC>
}

func.func @main(%arg0: tensor<1x4x4x32xf32>) -> tensor<1x4x4x32xf32> {
  %0 = cudnn.call @graph(%arg0) : (tensor<1x4x4x32xf32>) -> tensor<1x4x4x32xf32>
  return %0 : tensor<1x4x4x32xf32>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

cudnn.graph @graph(%arg0: !cudnn.tensor<1x32x4x4xf32, #NHWC>)
                       -> !cudnn.tensor<1x32x4x4xf32, #NHWC> {
  cudnn.return %arg0: !cudnn.tensor<1x32x4x4xf32, #NHWC>
}

func.func @main(%arg0: tensor<1x4x4x32xf32>) -> tensor<1x4x4x32xf32> {
  %0 = cudnn.call @graph(%arg0) : (tensor<1x4x4x32xf32>) -> tensor<1x4x4x32xf32>
  return %0 : tensor<1x4x4x32xf32>
}

// -----

cudnn.graph @convolution(
  %image: !cudnn.tensor<8x32x4x4xf32, NHWC>,
  %filter: !cudnn.tensor<32x32x1x1xf32, NHWC>
) -> !cudnn.tensor<8x32x4x4xf32, NHWC> {
  %0 = cudnn.convolution(%image, %filter) alpha=1.0 beta=0.0
         spatial_dim_count=2
         spatial_stride=[1,1]
         pre_padding=[1,1]
         post_padding=[1,1]
         dilation=[1,1]
    : (!cudnn.tensor<8x32x4x4xf32, NHWC>, !cudnn.tensor<32x32x1x1xf32, NHWC>)
    -> !cudnn.tensor<8x32x4x4xf32, NHWC>
  cudnn.return %0: !cudnn.tensor<8x32x4x4xf32, NHWC>
}

// CHECK: cudnn.graph @convolution {{.*}} {
// CHECK:   cudnn.convolution
// CHECK:     alpha = 1.000000e+00
// CHECK:     beta = 0.000000e+00
// CHECK:     spatial_dim_count = 2
// CHECK:     spatial_stride = [1, 1]
// CHECK:     pre_padding = [1, 1]
// CHECK:     post_padding = [1, 1]
// CHECK:     dilation = [1, 1]
// CHECK: }
