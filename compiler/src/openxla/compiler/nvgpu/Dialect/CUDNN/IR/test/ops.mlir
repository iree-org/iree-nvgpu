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
         spatial_dim_count = 2
         spatial_stride = [1,1]
         pre_padding = [1,1]
         post_padding = [1,1]
         dilation = [1,1]
    : (!cudnn.tensor<8x32x4x4xf32, NHWC>, !cudnn.tensor<32x32x1x1xf32, NHWC>)
    -> !cudnn.tensor<8x32x4x4xf32, NHWC>
  cudnn.return %0: !cudnn.tensor<8x32x4x4xf32, NHWC>
}

// CHECK: cudnn.graph @convolution
// CHECK: {
// CHECK:   cudnn.convolution
// CHECK:     alpha = 1.000000e+00
// CHECK:     beta = 0.000000e+00
// CHECK:     spatial_dim_count = 2
// CHECK:     spatial_stride = [1, 1]
// CHECK:     pre_padding = [1, 1]
// CHECK:     post_padding = [1, 1]
// CHECK:     dilation = [1, 1]
// CHECK: }

// -----

cudnn.graph @convolution(
  %image: !cudnn.tensor<8x32x4x4xf32, NHWC>,
  %filter: !cudnn.tensor<32x32x1x1xf32, NHWC>
) -> !cudnn.tensor<8x32x4x4xf32, NHWC> {
  %0 = cudnn.convolution(%image, %filter) alpha=1.0 beta=0.0
         spatial_dim_count = 2
         spatial_stride = [1,1]
         pre_padding = [1,1]
         post_padding = [1,1]
         dilation = [1,1]
         mode = CONVOLUTION
    : (!cudnn.tensor<8x32x4x4xf32, NHWC>, !cudnn.tensor<32x32x1x1xf32, NHWC>)
    -> !cudnn.tensor<8x32x4x4xf32, NHWC>
  cudnn.return %0: !cudnn.tensor<8x32x4x4xf32, NHWC>
}

// CHECK: cudnn.graph @convolution
// CHECK: {
// CHECK:   cudnn.convolution
// CHECK:     mode = CONVOLUTION
// CHECK: }

// -----

cudnn.graph @batch_norm_inference(
  %x: !cudnn.tensor<8x32x4x4xf32, NHWC>,
  %vec: !cudnn.tensor<1x32x1x1xf32, NHWC>,
  %scalar: !cudnn.tensor<1x1x1x1xf32, NHWC>
) -> !cudnn.tensor<8x32x4x4xf32, NHWC> {
  %0 = cudnn.batch_norm_inference(%x, %vec, %vec, %vec, %vec, %scalar)
    : (!cudnn.tensor<8x32x4x4xf32, NHWC>, !cudnn.tensor<1x32x1x1xf32, NHWC>,
       !cudnn.tensor<1x32x1x1xf32, NHWC>, !cudnn.tensor<1x32x1x1xf32, NHWC>,
       !cudnn.tensor<1x32x1x1xf32, NHWC>, !cudnn.tensor<1x1x1x1xf32, NHWC>)
    -> !cudnn.tensor<8x32x4x4xf32, NHWC>
  cudnn.return %0: !cudnn.tensor<8x32x4x4xf32, NHWC>
}

// CHECK: cudnn.graph @batch_norm_inference
// CHECK: {
// CHECK:   cudnn.batch_norm_inference
// CHECK: }

// -----

cudnn.graph @sqrt(%x: !cudnn.tensor<8x4x4xf32>) -> !cudnn.tensor<8x4x4xf32> {
  %0 = cudnn.sqrt(%x) : (!cudnn.tensor<8x4x4xf32>) -> !cudnn.tensor<8x4x4xf32>
  cudnn.return %0: !cudnn.tensor<8x4x4xf32>
}

// CHECK: cudnn.graph @sqrt
// CHECK: {
// CHECK:   cudnn.sqrt
// CHECK-NOT: alpha
// CHECK: }

// -----

cudnn.graph @sqrt(%x: !cudnn.tensor<8x4x4xf32>) -> !cudnn.tensor<8x4x4xf32> {
  %0 = cudnn.sqrt(%x) alpha=1.0 : (!cudnn.tensor<8x4x4xf32>)
                                -> !cudnn.tensor<8x4x4xf32>
  cudnn.return %0: !cudnn.tensor<8x4x4xf32>
}

// CHECK: cudnn.graph @sqrt
// CHECK: {
// CHECK:   cudnn.sqrt
// CHECK:     alpha = 1.000000e+00
// CHECK: }

// -----

cudnn.graph @add(
  %x: !cudnn.tensor<8x32x4x4xf32>,
  %b: !cudnn.tensor<8x32x4x4xf32>
) -> !cudnn.tensor<8x32x4x4xf32> {
  %0 = cudnn.add(%x, %b)
    : (!cudnn.tensor<8x32x4x4xf32>, !cudnn.tensor<8x32x4x4xf32>)
    -> !cudnn.tensor<8x32x4x4xf32>
  cudnn.return %0: !cudnn.tensor<8x32x4x4xf32>
}

// CHECK: cudnn.graph @add
// CHECK: {
// CHECK:   cudnn.add
// CHECK-NOT: alpha
// CHECK-NOT: alpha2
// CHECK: }

// -----

cudnn.graph @add(
  %x: !cudnn.tensor<8x32x4x4xf32>,
  %b: !cudnn.tensor<8x32x4x4xf32>
) -> !cudnn.tensor<8x32x4x4xf32> {
  %0 = cudnn.add(%x, %b) alpha=1.0 alpha2=1.0
    : (!cudnn.tensor<8x32x4x4xf32>, !cudnn.tensor<8x32x4x4xf32>)
    -> !cudnn.tensor<8x32x4x4xf32>
  cudnn.return %0: !cudnn.tensor<8x32x4x4xf32>
}

// CHECK: cudnn.graph @add
// CHECK: {
// CHECK:   cudnn.add
// CHECK:     alpha = 1.000000e+00
// CHECK:     alpha2 = 1.000000e+00
// CHECK: }

// -----

cudnn.graph @div(
  %x: !cudnn.tensor<8x32x4x4xf32>,
  %b: !cudnn.tensor<8x32x4x4xf32>
) -> !cudnn.tensor<8x32x4x4xf32> {
  %0 = cudnn.div(%x, %b) alpha=1.0 alpha2=1.0
    : (!cudnn.tensor<8x32x4x4xf32>, !cudnn.tensor<8x32x4x4xf32>)
    -> !cudnn.tensor<8x32x4x4xf32>
  cudnn.return %0: !cudnn.tensor<8x32x4x4xf32>
}

// CHECK: cudnn.graph @div
// CHECK: {
// CHECK:   cudnn.div
// CHECK:     alpha = 1.000000e+00
// CHECK:     alpha2 = 1.000000e+00
// CHECK: }

// -----

cudnn.graph @max(
  %x: !cudnn.tensor<8x32x4x4xf32>,
  %b: !cudnn.tensor<8x32x4x4xf32>
) -> !cudnn.tensor<8x32x4x4xf32> {
  %0 = cudnn.max(%x, %b) alpha=1.0 alpha2=1.0
    : (!cudnn.tensor<8x32x4x4xf32>, !cudnn.tensor<8x32x4x4xf32>)
    -> !cudnn.tensor<8x32x4x4xf32>
  cudnn.return %0: !cudnn.tensor<8x32x4x4xf32>
}

// CHECK: cudnn.graph @max
// CHECK: {
// CHECK:   cudnn.max
// CHECK:     alpha = 1.000000e+00
// CHECK:     alpha2 = 1.000000e+00
// CHECK: }

// -----

cudnn.graph @mul(
  %x: !cudnn.tensor<8x32x4x4xf32>,
  %b: !cudnn.tensor<8x32x4x4xf32>
) -> !cudnn.tensor<8x32x4x4xf32> {
  %0 = cudnn.mul(%x, %b) alpha=1.0 alpha2=1.0
    : (!cudnn.tensor<8x32x4x4xf32>, !cudnn.tensor<8x32x4x4xf32>)
    -> !cudnn.tensor<8x32x4x4xf32>
  cudnn.return %0: !cudnn.tensor<8x32x4x4xf32>
}

// CHECK: cudnn.graph @mul
// CHECK: {
// CHECK:   cudnn.mul
// CHECK:     alpha = 1.000000e+00
// CHECK:     alpha2 = 1.000000e+00
// CHECK: }

// -----

cudnn.graph @sub(
  %x: !cudnn.tensor<8x32x4x4xf32>,
  %b: !cudnn.tensor<8x32x4x4xf32>
) -> !cudnn.tensor<8x32x4x4xf32> {
  %0 = cudnn.sub(%x, %b) alpha=1.0 alpha2=1.0
    : (!cudnn.tensor<8x32x4x4xf32>, !cudnn.tensor<8x32x4x4xf32>)
    -> !cudnn.tensor<8x32x4x4xf32>
  cudnn.return %0: !cudnn.tensor<8x32x4x4xf32>
}

// CHECK: cudnn.graph @sub
// CHECK: {
// CHECK:   cudnn.sub
// CHECK:     alpha = 1.000000e+00
// CHECK:     alpha2 = 1.000000e+00
// CHECK: }

// -----

cudnn.graph @bias(
  %x: !cudnn.tensor<8x32x4x4xf32>,
  %b: !cudnn.tensor<1x32x1x1xf32>
) -> !cudnn.tensor<8x32x4x4xf32> {
  %0 = cudnn.bias(%x, %b)
    : (!cudnn.tensor<8x32x4x4xf32>, !cudnn.tensor<1x32x1x1xf32>)
    -> !cudnn.tensor<8x32x4x4xf32>
  cudnn.return %0: !cudnn.tensor<8x32x4x4xf32>
}

// CHECK: cudnn.graph @bias
// CHECK: {
// CHECK:   cudnn.bias
// CHECK: }
