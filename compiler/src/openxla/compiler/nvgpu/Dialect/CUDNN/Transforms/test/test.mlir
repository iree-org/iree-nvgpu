cudnn.graph @graph(%arg: !cudnn.tensor<1x4x8xf32>) -> !cudnn.tensor<1x4x8xf32> {
  cudnn.return %arg: !cudnn.tensor<1x4x8xf32>
}

func.func @main(%handle: !cudnn.handle,
                %arg: tensor<1x4x8xf32>) -> (tensor<1x4x8xf32>, tensor<1x4x8xf32>) {
  %0 = cudnn.call handle(%handle) @graph(%arg)
       : (tensor<1x4x8xf32>) -> tensor<1x4x8xf32>

  %1 = cudnn.call handle(%handle) @graph(%arg)
       : (tensor<1x4x8xf32>) -> tensor<1x4x8xf32>

  return %0, %1 : tensor<1x4x8xf32>, tensor<1x4x8xf32>
}
