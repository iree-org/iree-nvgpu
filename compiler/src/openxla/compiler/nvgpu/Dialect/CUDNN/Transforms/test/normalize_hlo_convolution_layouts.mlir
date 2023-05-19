// RUN: iree-opt %s --split-input-file --iree-plugin=openxla-cudnn             \
// RUN:     --openxla-nvgpu-normalize-hlo-convolution-layouts="tensor-layout=NHWC kernel-layout=KCHW" \
// RUN: | FileCheck %s --check-prefix=CHECK-KCHW

// RUN: iree-opt %s --split-input-file --iree-plugin=openxla-cudnn             \
// RUN:     --openxla-nvgpu-normalize-hlo-convolution-layouts="tensor-layout=NHWC kernel-layout=KHWC" \
// RUN: | FileCheck %s --check-prefix=CHECK-KHWC


func.func @conv_NHWC_HWCK_NHWC(%lhs : tensor<1x56x56x64xf32>,
    %rhs : tensor<1x1x64x256xf32>) -> tensor<1x56x56x256xf32> {
  %0 = stablehlo.convolution(%lhs, %rhs)
      dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {
      batch_group_count = 1 : i64, feature_group_count = 1 : i64}
      : (tensor<1x56x56x64xf32>, tensor<1x1x64x256xf32>)
      -> tensor<1x56x56x256xf32>
  return %0 : tensor<1x56x56x256xf32>
}

// CHECK-KCHW-LABEL: @conv_NHWC_HWCK_NHWC
// CHECK-KCHW-SAME:      %[[ARG0:.*]]: tensor<1x56x56x64xf32>, %[[ARG1:.*]]: tensor<1x1x64x256xf32>
// CHECK-KCHW:       %[[TRANSPOSE:.*]] = stablehlo.transpose %[[ARG1]], dims = [3, 2, 0, 1]
// CHECK-KCHW:       %[[CONVOLUTION:.*]] = stablehlo.convolution(%[[ARG0]], %[[TRANSPOSE]])
// CHECK-KCHW-SAME:      dim_numbers = [b, 0, 1, f]x[o, i, 0, 1]->[b, 0, 1, f]
// CHECK-KCHW-SAME:      window = {stride = [1, 1], pad = {{\[\[}}0, 0], [0, 0{{\]\]}}, rhs_dilate = [1, 1]}
// CHECK-KCHW-SAME:      batch_group_count = 1
// CHECK-KCHW-SAME:      feature_group_count = 1
// CHECK-KCHW:       return %[[CONVOLUTION]]

// CHECK-KHWC-LABEL: @conv_NHWC_HWCK_NHWC
// CHECK-KHWC-SAME:      %[[ARG0:.*]]: tensor<1x56x56x64xf32>, %[[ARG1:.*]]: tensor<1x1x64x256xf32>
// CHECK-KHWC:       %[[TRANSPOSE:.*]] = stablehlo.transpose %[[ARG1]], dims = [3, 0, 1, 2]
// CHECK-KHWC:       %[[CONVOLUTION:.*]] = stablehlo.convolution(%[[ARG0]], %[[TRANSPOSE]])
// CHECK-KHWC-SAME:      dim_numbers = [b, 0, 1, f]x[o, 0, 1, i]->[b, 0, 1, f]
// CHECK-KHWC-SAME:      window = {stride = [1, 1], pad = {{\[\[}}0, 0], [0, 0{{\]\]}}, rhs_dilate = [1, 1]}
// CHECK-KHWC-SAME:      batch_group_count = 1
// CHECK-KHWC-SAME:      feature_group_count = 1
// CHECK-KHWC:       return %[[CONVOLUTION]]

// -----

func.func @conv_HCNW_HWCK_HCNW(%lhs : tensor<56x64x1x56xf32>,
    %rhs : tensor<1x1x64x256xf32>) -> tensor<56x256x1x56xf32> {
  %0 = stablehlo.convolution(%lhs, %rhs)
      dim_numbers = [0, f, b, 1]x[0, 1, i, o]->[0, f, b, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {
      batch_group_count = 1 : i64,
      feature_group_count = 1 : i64}
      : (tensor<56x64x1x56xf32>, tensor<1x1x64x256xf32>)
      -> tensor<56x256x1x56xf32>
  return %0 : tensor<56x256x1x56xf32>
}

// CHECK-KCHW-LABEL: @conv_HCNW_HWCK_HCNW
// CHECK-KCHW-SAME:      %[[ARG0:.*]]: tensor<56x64x1x56xf32>, %[[ARG1:.*]]: tensor<1x1x64x256xf32>
// CHECK-KCHW:       %[[TRANSPOSE:.*]] = stablehlo.transpose %[[ARG0]], dims = [2, 0, 3, 1]
// CHECK-KCHW:       %[[TRANSPOSE_0:.*]] = stablehlo.transpose %[[ARG1]], dims = [3, 2, 0, 1]
// CHECK-KCHW:       %[[CONVOLUTION:.*]] = stablehlo.convolution(%[[TRANSPOSE]], %[[TRANSPOSE_0]])
// CHECK-KCHW-SAME:      dim_numbers = [b, 0, 1, f]x[o, i, 0, 1]->[b, 0, 1, f]
// CHECK-KCHW-SAME:      window = {stride = [1, 1], pad = {{\[\[}}0, 0], [0, 0{{\]\]}}, rhs_dilate = [1, 1]}
// CHECK-KCHW-SAME:      batch_group_count = 1
// CHECK-KCHW-SAME:      feature_group_count = 1
// CHECK-KCHW:       %[[TRANSPOSE_1:.*]] = stablehlo.transpose %[[CONVOLUTION]], dims = [1, 3, 0, 2]
// CHECK-KCHW:       return %[[TRANSPOSE_1]]

// CHECK-KHWC-LABEL: @conv_HCNW_HWCK_HCNW
// CHECK-KHWC-SAME:      %[[ARG0_0:.*]]: tensor<56x64x1x56xf32>, %[[ARG1_0:.*]]: tensor<1x1x64x256xf32>
// CHECK-KHWC:       %[[TRANSPOSE_0:.*]] = stablehlo.transpose %[[ARG0_0]], dims = [2, 0, 3, 1]
// CHECK-KHWC:       %[[TRANSPOSE_1:.*]] = stablehlo.transpose %[[ARG1_0]], dims = [3, 0, 1, 2]
// CHECK-KHWC:       %[[CONVOLUTION_0:.*]] = stablehlo.convolution(%[[TRANSPOSE_0]], %[[TRANSPOSE_1]])
// CHECK-KHWC-SAME:      dim_numbers = [b, 0, 1, f]x[o, 0, 1, i]->[b, 0, 1, f]
// CHECK-KHWC-SAME:      window = {stride = [1, 1], pad = {{\[\[}}0, 0], [0, 0{{\]\]}}, rhs_dilate = [1, 1]}
// CHECK-KHWC-SAME:      batch_group_count = 1
// CHECK-KHWC-SAME:      feature_group_count = 1
// CHECK-KHWC:       %[[TRANSPOSE_2:.*]] = stablehlo.transpose %[[CONVOLUTION_0]], dims = [1, 3, 0, 2]
// CHECK-KHWC:       return %[[TRANSPOSE_2]]

// -----

func.func @conv_HCNW_KCHW_HCNW(%lhs : tensor<56x64x1x56xf32>,
    %rhs : tensor<256x64x1x1xf32>) -> tensor<56x256x1x56xf32> {
  %0 = stablehlo.convolution(%lhs, %rhs)
      dim_numbers = [0, f, b, 1]x[o, i, 0, 1]->[0, f, b, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {
      batch_group_count = 1 : i64,
      feature_group_count = 1 : i64}
      : (tensor<56x64x1x56xf32>, tensor<256x64x1x1xf32>)
      -> tensor<56x256x1x56xf32>
  return %0 : tensor<56x256x1x56xf32>
}

// CHECK-KCHW-LABEL: @conv_HCNW_KCHW_HCNW
// CHECK-KCHW-SAME:      %[[ARG0:.*]]: tensor<56x64x1x56xf32>, %[[ARG1:.*]]: tensor<256x64x1x1xf32>
// CHECK-KCHW:       %[[TRANSPOSE:.*]] = stablehlo.transpose %[[ARG0]], dims = [2, 0, 3, 1]
// CHECK-KCHW:       %[[CONVOLUTION:.*]] = stablehlo.convolution(%[[TRANSPOSE]], %[[ARG1]])
// CHECK-KCHW-SAME:      dim_numbers = [b, 0, 1, f]x[o, i, 0, 1]->[b, 0, 1, f]
// CHECK-KCHW-SAME:      window = {stride = [1, 1], pad = {{\[\[}}0, 0], [0, 0{{\]\]}}, rhs_dilate = [1, 1]}
// CHECK-KCHW-SAME:      batch_group_count = 1
// CHECK-KCHW-SAME:      feature_group_count = 1
// CHECK-KCHW:       %[[TRANSPOSE_0:.*]] = stablehlo.transpose %[[CONVOLUTION]], dims = [1, 3, 0, 2]
// CHECK-KCHW:       return %[[TRANSPOSE_0]]

// CHECK-KHWC-LABEL: @conv_HCNW_KCHW_HCNW
// CHECK-KHWC-SAME:      %[[ARG0_1:.*]]: tensor<56x64x1x56xf32>, %[[ARG1_1:.*]]: tensor<256x64x1x1xf32>
// CHECK-KHWC:       %[[TRANSPOSE_3:.*]] = stablehlo.transpose %[[ARG0_1]], dims = [2, 0, 3, 1]
// CHECK-KHWC:       %[[TRANSPOSE_4:.*]] = stablehlo.transpose %[[ARG1_1]], dims = [0, 2, 3, 1]
// CHECK-KHWC:       %[[CONVOLUTION_1:.*]] = stablehlo.convolution(%[[TRANSPOSE_3]], %[[TRANSPOSE_4]])
// CHECK-KHWC-SAME:      dim_numbers = [b, 0, 1, f]x[o, 0, 1, i]->[b, 0, 1, f],
// CHECK-KHWC-SAME:      window = {stride = [1, 1], pad = {{\[\[}}0, 0], [0, 0{{\]\]}}, rhs_dilate = [1, 1]}
// CHECK-KHWC-SAME:      batch_group_count = 1
// CHECK-KHWC-SAME:      feature_group_count = 1
// CHECK-KHWC:       %[[TRANSPOSE_5:.*]] = stablehlo.transpose %[[CONVOLUTION_1]], dims = [1, 3, 0, 2]
// CHECK-KHWC:       return %[[TRANSPOSE_5]]

// -----

func.func @chained_conv_HCNW_KCHW_HCNW(%arg : tensor<56x64x1x56xf32>,
    %f0 : tensor<64x64x1x1xf32>, %f1 : tensor<256x64x1x1xf32>)
    -> tensor<56x256x1x56xf32> {
  %0 = stablehlo.convolution(%arg, %f0)
      dim_numbers = [0, f, b, 1]x[o, i, 0, 1]->[0, f, b, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {
      batch_group_count = 1 : i64,
      feature_group_count = 1 : i64}
      : (tensor<56x64x1x56xf32>, tensor<64x64x1x1xf32>)
      -> tensor<56x64x1x56xf32>
  %1 = stablehlo.convolution(%0, %f1)
      dim_numbers = [0, f, b, 1]x[o, i, 0, 1]->[0, f, b, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {
      batch_group_count = 1 : i64,
      feature_group_count = 1 : i64}
      : (tensor<56x64x1x56xf32>, tensor<256x64x1x1xf32>)
      -> tensor<56x256x1x56xf32>
  return %1 : tensor<56x256x1x56xf32>
}

// CHECK-KCHW-LABEL: @chained_conv_HCNW_KCHW_HCNW
// CHECK-KCHW-SAME:      %[[ARG0:.*]]: tensor<56x64x1x56xf32>, %[[ARG1:.*]]: tensor<64x64x1x1xf32>, %[[ARG2:.*]]: tensor<256x64x1x1xf32>
// CHECK-KCHW:       %[[TRANSPOSE:.*]] = stablehlo.transpose %[[ARG0]], dims = [2, 0, 3, 1]
// CHECK-KCHW:       %[[CONVOLUTION:.*]] = stablehlo.convolution(%[[TRANSPOSE]], %[[ARG1]])
// CHECK-KCHW-SAME:      dim_numbers = [b, 0, 1, f]x[o, i, 0, 1]->[b, 0, 1, f]
// CHECK-KCHW-SAME:      window = {stride = [1, 1], pad = {{\[\[}}0, 0], [0, 0{{\]\]}}, rhs_dilate = [1, 1]}
// CHECK-KCHW-SAME:      batch_group_count = 1
// CHECK-KCHW-SAME:      feature_group_count = 1
// CHECK-KCHW:       %[[CONVOLUTION_0:.*]] = stablehlo.convolution(%[[CONVOLUTION]], %[[ARG2]])
// CHECK-KCHW-SAME:      dim_numbers = [b, 0, 1, f]x[o, i, 0, 1]->[b, 0, 1, f]
// CHECK-KCHW-SAME:      window = {stride = [1, 1], pad = {{\[\[}}0, 0], [0, 0{{\]\]}}, rhs_dilate = [1, 1]}
// CHECK-KCHW-SAME:      batch_group_count = 1
// CHECK-KCHW-SAME:      feature_group_count = 1
// CHECK-KCHW:       %[[TRANSPOSE_0:.*]] = stablehlo.transpose %[[CONVOLUTION_0]], dims = [1, 3, 0, 2]
// CHECK-KCHW:       return %[[TRANSPOSE_0]]

// CHECK-KHWC-LABEL: @chained_conv_HCNW_KCHW_HCNW
// CHECK-KHWC-SAME:      %[[ARG0_2:.*]]: tensor<56x64x1x56xf32>, %[[ARG1_2:.*]]: tensor<64x64x1x1xf32>, %[[ARG2:.*]]: tensor<256x64x1x1xf32>
// CHECK-KHWC:       %[[TRANSPOSE_6:.*]] = stablehlo.transpose %[[ARG0_2]], dims = [2, 0, 3, 1]
// CHECK-KHWC:       %[[TRANSPOSE_7:.*]] = stablehlo.transpose %[[ARG1_2]], dims = [0, 2, 3, 1]
// CHECK-KHWC:       %[[CONVOLUTION_2:.*]] = stablehlo.convolution(%[[TRANSPOSE_6]], %[[TRANSPOSE_7]])
// CHECK-KHWC-SAME:      dim_numbers = [b, 0, 1, f]x[o, 0, 1, i]->[b, 0, 1, f]
// CHECK-KHWC-SAME:      window = {stride = [1, 1], pad = {{\[\[}}0, 0], [0, 0{{\]\]}}, rhs_dilate = [1, 1]}
// CHECK-KHWC-SAME:      batch_group_count = 1
// CHECK-KHWC-SAME:      feature_group_count = 1
// CHECK-KHWC:       %[[TRANSPOSE_8:.*]] = stablehlo.transpose %[[ARG2]], dims = [0, 2, 3, 1]
// CHECK-KHWC:       %[[CONVOLUTION_3:.*]] = stablehlo.convolution(%[[CONVOLUTION_2]], %[[TRANSPOSE_8]])
// CHECK-KHWC-SAME:      dim_numbers = [b, 0, 1, f]x[o, 0, 1, i]->[b, 0, 1, f]
// CHECK-KHWC-SAME:      window = {stride = [1, 1], pad = {{\[\[}}0, 0], [0, 0{{\]\]}}, rhs_dilate = [1, 1]}
// CHECK-KHWC-SAME:      batch_group_count = 1
// CHECK-KHWC-SAME:      feature_group_count = 1
// CHECK-KHWC:       %[[TRANSPOSE_9:.*]] = stablehlo.transpose %[[CONVOLUTION_3]], dims = [1, 3, 0, 2]
// CHECK-KHWC:       return %[[TRANSPOSE_9]]
