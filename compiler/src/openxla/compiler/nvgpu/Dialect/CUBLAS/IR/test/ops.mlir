// RUN: iree-opt %s --iree-plugin=openxla-cublas --split-input-file            \
// RUN:   | iree-opt --iree-plugin=openxla-cublas --split-input-file           \
// RUN:   | FileCheck %s

func.func @main(%arg0: !hal.device) -> !cublas.handle {
  %0 = cublas.handle(%arg0) : !cublas.handle
  return %0 : !cublas.handle
}

// CHECK: func @main(%[[ARG0:.*]]: !hal.device) -> !cublas.handle {
// CHECK:   cublas.handle(%[[ARG0]]) : !cublas.handle
// CHECK: }

// -----

func.func @gemm(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>) {
  %0 = cublas.gemm(%arg0, %arg1) : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  return
}

// CHECK: func @gemm(%[[ARG0:.*]]: tensor<4x4xf32>, %[[ARG1:.*]]: tensor<4x4xf32>)
// CHECK:   cublas.gemm(%[[ARG0]], %[[ARG1]])

// -----

func.func @inplace_gemm(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>, %arg2: tensor<4x4xf32>) {
  %0 = cublas.gemm(%arg0, %arg1, %arg2) alpha = 0.5 beta = 0.5 : (tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>) -> %arg2
  return
}

// CHECK: func @inplace_gemm(%[[ARG0:.*]]: tensor<4x4xf32>, %[[ARG1:.*]]: tensor<4x4xf32>, %[[ARG2:.*]]: tensor<4x4xf32>)
// CHECK:   cublas.gemm(%[[ARG0]], %[[ARG1]], %[[ARG2]]) alpha = 5.000000e-01 beta = 5.000000e-01 {{.*}} -> %[[ARG2]]

// -----

func.func @gemm_transposed(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>) {
  %0 = cublas.gemm(%arg0, %arg1) transa = T transb = T : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  return
}

// CHECK: func @gemm_transposed(%[[ARG0:.*]]: tensor<4x4xf32>, %[[ARG1:.*]]: tensor<4x4xf32>)
// CHECK:   cublas.gemm(%[[ARG0]], %[[ARG1]]) transa = T transb = T

// -----

func.func @gemm_algo(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>) {
  %0 = cublas.gemm(%arg0, %arg1) algo = ALGO0 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  return
}

// CHECK: func @gemm_algo(%[[ARG0:.*]]: tensor<4x4xf32>, %[[ARG1:.*]]: tensor<4x4xf32>)
// CHECK:   cublas.gemm(%[[ARG0]], %[[ARG1]]) algo = ALGO0

// -----

func.func @gemm_compute_type(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>) {
  %0 = cublas.gemm(%arg0, %arg1) compute_type = R_64F : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  return
}

// CHECK: func @gemm_compute_type(%[[ARG0:.*]]: tensor<4x4xf32>, %[[ARG1:.*]]: tensor<4x4xf32>)
// CHECK:   cublas.gemm(%[[ARG0]], %[[ARG1]]) compute_type = R_64F
