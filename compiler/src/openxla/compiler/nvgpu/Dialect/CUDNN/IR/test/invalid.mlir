// RUN: iree-opt %s --iree-plugin=openxla_nvgpu --split-input-file --verify-diagnostics

// expected-error @+1 {{requires all arguments to be non-opaque cuDNN tensors}}
cudnn.graph @g(%arg0: tensor<?x?x?xf32>) -> tensor<?x?x?xf32> {
  cudnn.return %arg0: tensor<?x?x?xf32>
}

// -----

// expected-error @+1 {{requires all arguments to be non-opaque cuDNN tensors}}
cudnn.graph @g(%arg0: !cudnn.tensor) -> !cudnn.tensor {
  cudnn.return %arg0: !cudnn.tensor
}

// -----

// expected-error @+1 {{op requires exactly one cuDNN tensor result}}
cudnn.graph @g(%arg0: !cudnn.tensor<?x?x?xf32>) {
  cudnn.return
}

// -----

cudnn.graph @graph(%arg0: !cudnn.tensor<1x32x4x4xf32, NHWC>)
                       -> !cudnn.tensor<1x32x4x4xf32, NHWC> {
  cudnn.return %arg0: !cudnn.tensor<1x32x4x4xf32, NHWC>
}

func.func @main(%arg0: tensor<1x32x4x4xf32>) -> tensor<1x32x4x4xf32> {
  %device = hal.ex.shared_device : !hal.device
  %handle = cudnn.handle(%device) : !cudnn.handle
  // expected-error @+1 {{argument #0 shape [1, 32, 4, 4] doesn't match the expected shape [1, 4, 4, 32]}}
  %0 = cudnn.call handle(%handle) @graph(%arg0) : (tensor<1x32x4x4xf32>) -> tensor<1x32x4x4xf32>
  return %0 : tensor<1x32x4x4xf32>
}
