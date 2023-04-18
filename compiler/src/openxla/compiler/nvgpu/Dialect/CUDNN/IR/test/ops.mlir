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
