// RUN: iree-opt %s --iree-plugin=openxla-cublas --split-input-file --verify-diagnostics

func.func @main(%arg0: tensor<4x4xf32>) {
  // expected-error @+1 {{op result #0 must be 2D tensor of any supported cuBLAS data type values, but got 'tensor<4x4x4xf32>'}}
  cublas.gemm(%arg0, %arg0) : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4x4xf32>
  return
}

// -----

func.func @main(%arg0: tensor<4x4xf32>) {
  // expected-error @+1 {{op must have two or three arguments}}
  cublas.gemm(%arg0) : (tensor<4x4xf32>) -> tensor<4x4xf32>
  return
}

// -----

func.func @main(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>) {
  // expected-error @+1 {{op must have exactly one result}}
  cublas.gemm(%arg0, %arg1) : (tensor<4x4xf32>, tensor<4x4xf32>) -> (tensor<4x4xf32>, tensor<4x4xf32>)
  return
}

// -----

func.func @main(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>, %arg2: tensor<4x4xf32>) {
  // expected-error @+1 {{op only third argument can be used as a tied operand}}
  cublas.gemm(%arg0, %arg1, %arg2) : (tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>) -> %arg1
  return
}

// -----

func.func @main(%arg0: tensor<4x4xf32>) {
  // expected-error @+1 {{op without argument C must have beta equal 0.0}}
  cublas.gemm(%arg0, %arg0) beta = 0.5 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  return
}
