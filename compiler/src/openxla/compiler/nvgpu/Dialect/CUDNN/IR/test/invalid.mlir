// RUN: iree-opt --iree-plugin=openxla_nvgpu --split-input-file --verify-diagnostics %s

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
