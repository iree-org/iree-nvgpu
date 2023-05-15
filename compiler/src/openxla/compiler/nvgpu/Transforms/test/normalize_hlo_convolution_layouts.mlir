// RUN: iree-opt %s --split-input-file --iree-plugin=openxla_nvgpu             \
// RUN:     --openxla-nvgpu-normalize-hlo-convolution-layouts                  \
// RUN: | FileCheck %s


// CHECK-LABEL: @conv_NHWC_KHWC_NHWC
// CHECK-SAME:  %[[ARG0:.*]]: tensor<1x56x56x64xf32>, %[[ARG1:.*]]: tensor<1x1x64x256xf32>
func.func @conv_NHWC_KHWC_NHWC(%lhs : tensor<1x56x56x64xf32>,
    %rhs : tensor<1x1x64x256xf32>) -> tensor<1x56x56x256xf32> {
  // CHECK:      %[[TRANSPOSE:.*]] = stablehlo.transpose %[[ARG1]], dims = [3, 2, 0, 1]
  // CHECK:      %[[CONVOLUTION:.*]] = stablehlo.convolution(%[[ARG0]], %[[TRANSPOSE]])
  // CHECK-SAME:     dim_numbers = [b, 0, 1, f]x[o, i, 0, 1]->[b, 0, 1, f]
  // CHECK-SAME:     window = {stride = [1, 1], pad = {{\[\[}}0, 0], [0, 0{{\]\]}}, rhs_dilate = [1, 1]}
  // CHECK-SAME:     batch_group_count = 1
  // CHECK-SAME:     feature_group_count = 1
  // CHECK:      return %[[CONVOLUTION]]
  %0 = stablehlo.convolution(%lhs, %rhs)
      dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {
      batch_group_count = 1 : i64, feature_group_count = 1 : i64}
      : (tensor<1x56x56x64xf32>, tensor<1x1x64x256xf32>)
      -> tensor<1x56x56x256xf32>
  return %0 : tensor<1x56x56x256xf32>
}

// -----

// CHECK-LABEL: @conv_HCNW_KHWC_HCNW
// CHECK-SAME:  %[[ARG0:.*]]: tensor<56x64x1x56xf32>, %[[ARG1:.*]]: tensor<1x1x64x256xf32>
func.func @conv_HCNW_KHWC_HCNW(%lhs : tensor<56x64x1x56xf32>,
    %rhs : tensor<1x1x64x256xf32>) -> tensor<56x256x1x56xf32> {
  // CHECK:      %[[TRANSPOSE:.*]] = stablehlo.transpose %[[ARG0]], dims = [2, 0, 3, 1]
  // CHECK:      %[[TRANSPOSE_0:.*]] = stablehlo.transpose %[[ARG1]], dims = [3, 2, 0, 1]
  // CHECK:      %[[CONVOLUTION:.*]] = stablehlo.convolution(%[[TRANSPOSE]], %[[TRANSPOSE_0]])
  // CHECK-SAME:     dim_numbers = [b, 0, 1, f]x[o, i, 0, 1]->[b, 0, 1, f]
  // CHECK-SAME:     window = {stride = [1, 1], pad = {{\[\[}}0, 0], [0, 0{{\]\]}}, rhs_dilate = [1, 1]}
  // CHECK-SAME:     batch_group_count = 1
  // CHECK-SAME:     feature_group_count = 1
  // CHECK:      %[[TRANSPOSE_1:.*]] = stablehlo.transpose %[[CONVOLUTION]], dims = [1, 3, 0, 2]
  // CHECK:      return %[[TRANSPOSE_1]]
  %0 = stablehlo.convolution(%lhs, %rhs)
      dim_numbers = [0, f, b, 1]x[0, 1, i, o]->[0, f, b, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {
      batch_group_count = 1 : i64,
      feature_group_count = 1 : i64}
      : (tensor<56x64x1x56xf32>, tensor<1x1x64x256xf32>)
      -> tensor<56x256x1x56xf32>
  return %0 : tensor<56x256x1x56xf32>
}

// -----

// CHECK-LABEL: @conv_HCNW_KCHW_HCNW
// CHECK-SAME:  %[[ARG0:.*]]: tensor<56x64x1x56xf32>, %[[ARG1:.*]]: tensor<256x64x1x1xf32>
func.func @conv_HCNW_KCHW_HCNW(%lhs : tensor<56x64x1x56xf32>,
    %rhs : tensor<256x64x1x1xf32>) -> tensor<56x256x1x56xf32> {
  // CHECK:      %[[TRANSPOSE:.*]] = stablehlo.transpose %[[ARG0]], dims = [2, 0, 3, 1]
  // CHECK:      %[[CONVOLUTION:.*]] = stablehlo.convolution(%[[TRANSPOSE]], %[[ARG1]])
  // CHECK-SAME:     dim_numbers = [b, 0, 1, f]x[o, i, 0, 1]->[b, 0, 1, f]
  // CHECK-SAME:     window = {stride = [1, 1], pad = {{\[\[}}0, 0], [0, 0{{\]\]}}, rhs_dilate = [1, 1]}
  // CHECK-SAME:     batch_group_count = 1
  // CHECK-SAME:     feature_group_count = 1
  // CHECK:      %[[TRANSPOSE_0:.*]] = stablehlo.transpose %[[CONVOLUTION]], dims = [1, 3, 0, 2]
  // CHECK:      return %[[TRANSPOSE_0]]
  %0 = stablehlo.convolution(%lhs, %rhs)
      dim_numbers = [0, f, b, 1]x[o, i, 0, 1]->[0, f, b, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {
      batch_group_count = 1 : i64,
      feature_group_count = 1 : i64}
      : (tensor<56x64x1x56xf32>, tensor<256x64x1x1xf32>)
      -> tensor<56x256x1x56xf32>
  return %0 : tensor<56x256x1x56xf32>
}
