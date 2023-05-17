// RUN: iree-opt %s --split-input-file --iree-plugin=openxla_nvgpu             \
// RUN:     --openxla-nvgpu-convert-hlo-to-cudnn                               \
// RUN: | FileCheck %s


// CHECK:      cudnn.graph @stablehlo.clamp(
// CHECK-SAME:   %[[ARGUMENT:.*]]: !cudnn.tensor<1x16x32x8xf32, NCHW>
// CHECK-SAME: ) -> !cudnn.tensor<1x16x32x8xf32, NCHW> {
// CHECK:        %[[RESULT:.*]] = cudnn.relu(%[[ARGUMENT]]) type = f32 lower_clip = 0.000000e+00
// CHECK-SAME      : (!cudnn.tensor<1x16x32x8xf32, NCHW>) -> !cudnn.tensor<1x16x32x8xf32, NCHW>
// CHECK:        cudnn.return %[[RESULT]]

// CHECK: @test_relu(%[[ARGUMENT]]: tensor<1x16x32x8xf32>)
func.func @test_relu(%argument: tensor<1x16x32x8xf32>) -> tensor<1x16x32x8xf32> {
  // CHECK: %[[DEVICE:.*]] = hal.ex.shared_device
  // CHECK: %[[HANDLE:.*]] = cudnn.handle(%[[DEVICE]])
  // CHECK: %[[CALL:.*]] = cudnn.call handle(%[[HANDLE]]) @stablehlo.clamp(%[[ARGUMENT]])
  // CHECK: return %[[CALL]]
  %min = stablehlo.constant dense<0.0> : tensor<1x16x32x8xf32>
  %max = stablehlo.constant dense<0xFFFFFFFF> : tensor<1x16x32x8xf32>
  %result = stablehlo.clamp %min, %argument, %max : tensor<1x16x32x8xf32>
  return %result : tensor<1x16x32x8xf32>
}

// -----

// CHECK:      cudnn.graph @stablehlo.convolution(
// CHECK-SAME:   %[[INPUT:.*]]: !cudnn.tensor<100x32x26x26xf32, NCHW>,
// CHECK-SAME:   %[[FILTER:.*]]: !cudnn.tensor<1x32x3x3xf32, NCHW>
// CHECK-SAME: ) -> !cudnn.tensor<100x1x28x28xf32, NCHW> {
// CHECK:        %[[RESULT:.*]] = cudnn.convolution(%[[INPUT]], %[[FILTER]])
// CHECK-SAME:   alpha = 1.000000e+00 beta = 0.000000e+00
// CHECK-SAME:   spatial_dim_count = 2
// CHECK-SAME:   spatial_stride = [1, 1]
// CHECK-SAME:   pre_padding = [2, 2]
// CHECK-SAME:   post_padding = [2, 2]
// CHECK-SAME:   dilation = [1, 1] : (
// CHECK-SAME:       !cudnn.tensor<100x32x26x26xf32, NCHW>,
// CHECK-SAME:       !cudnn.tensor<1x32x3x3xf32, NCHW>
// CHECK-SAME:     ) -> !cudnn.tensor<100x1x28x28xf32, NCHW>
// CHECK:        cudnn.return %[[RESULT]]

// CHECK:      @test_conv(
// CHECK-SAME:   %[[INPUT]]: tensor<100x32x26x26xf32>,
// CHECK-SAME:   %[[FILTER]]: tensor<1x32x3x3xf32>
// CHECK-SAME: ) -> tensor<100x1x28x28xf32> {
func.func @test_conv(
  %input : tensor<100x32x26x26xf32>,
  %filter : tensor<1x32x3x3xf32>
) -> tensor<100x1x28x28xf32> {
  // CHECK: %[[DEVICE_0:.*]] = hal.ex.shared_device
  // CHECK: %[[HANDLE_0:.*]] = cudnn.handle(%[[DEVICE_0]])
  // CHECK: %[[CALL_0:.*]] = cudnn.call handle(%[[HANDLE_0]]) @stablehlo.convolution(%[[INPUT]], %[[FILTER]])
  // CHECK: return %[[CALL_0]]
  %result = "stablehlo.convolution"(%input, %filter) {
    batch_group_count = 1 : i64,
    dimension_numbers = #stablehlo.conv<raw
      input_batch_dimension = 0,
      input_feature_dimension = 1,
      input_spatial_dimensions = [2, 3],
      kernel_input_feature_dimension = 1,
      kernel_output_feature_dimension = 0,
      kernel_spatial_dimensions = [2, 3],
      output_batch_dimension = 0,
      output_feature_dimension = 1,
      output_spatial_dimensions = [2, 3]
    >,
    feature_group_count = 1 : i64,
    lhs_dilation = dense<1> : tensor<2xi64>,
    padding = dense<2> : tensor<2x2xi64>,
    rhs_dilation = dense<1> : tensor<2xi64>,
    window_strides = dense<1> : tensor<2xi64>
  } : (tensor<100x32x26x26xf32>, tensor<1x32x3x3xf32>) -> tensor<100x1x28x28xf32>
  func.return %result : tensor<100x1x28x28xf32>
}

// -----

// CHECK:      cudnn.graph @stablehlo.clamp(
// CHECK-SAME:   %[[INPUT]]: !cudnn.tensor<100x32x26x26xf32, NHWC>,
// CHECK-SAME:   %[[FILTER:.*]]: !cudnn.tensor<1x32x3x3xf32, NHWC>
// CHECK-SAME: ) -> !cudnn.tensor<100x1x28x28xf32, NHWC> {
// CHECK:        %[[CONV:.*]] = cudnn.convolution(%[[INPUT]], %[[FILTER]])
// CHECK-SAME:     alpha = 1.000000e+00 beta = 0.000000e+00
// CHECK-SAME:     spatial_dim_count = 2
// CHECK-SAME:     spatial_stride = [1, 1]
// CHECK-SAME:     pre_padding = [2, 2]
// CHECK-SAME:     post_padding = [2, 2]
// CHECK-SAME:     dilation = [1, 1] : (
// CHECK-SAME:       !cudnn.tensor<100x32x26x26xf32, NHWC>,
// CHECK-SAME:       !cudnn.tensor<1x32x3x3xf32, NHWC>
// CHECK-SAME:     ) -> !cudnn.tensor<100x1x28x28xf32, NHWC>
// CHECK:        %[[RESULT:.*]] = cudnn.relu(%[[CONV]]) type = f32 lower_clip = 0.000000e+00
// CHECK-SAME:     : (!cudnn.tensor<100x1x28x28xf32, NHWC>) -> !cudnn.tensor<100x1x28x28xf32, NHWC>
// CHECK:        cudnn.return %[[RESULT]]

// CHECK:      @test_graph(
// CHECK-SAME:   %[[INPUT]]: tensor<100x26x26x32xf32>,
// CHECK-SAME:   %[[FILTER]]: tensor<1x3x3x32xf32>
// CHECK-SAME: ) -> tensor<100x28x28x1xf32> {
func.func @test_graph(
  %input : tensor<100x26x26x32xf32>,
  %filter : tensor<1x3x3x32xf32>
) -> tensor<100x28x28x1xf32> {
  // CHECK: %[[DEVICE:.*]] = hal.ex.shared_device
  // CHECK: %[[HANDLE:.*]] = cudnn.handle(%[[DEVICE]])
  // CHECK: %[[CALL:.*]] = cudnn.call handle(%[[HANDLE]]) @stablehlo.clamp(%[[INPUT]], %[[FILTER]])
  // CHECK: return %[[CALL_0]]
  %conv = "stablehlo.convolution"(%input, %filter) {
    batch_group_count = 1 : i64,
    dimension_numbers = #stablehlo.conv<raw
      input_batch_dimension = 0,
      input_feature_dimension = 3,
      input_spatial_dimensions = [1, 2],
      kernel_input_feature_dimension = 3,
      kernel_output_feature_dimension = 0,
      kernel_spatial_dimensions = [1, 2],
      output_batch_dimension = 0,
      output_feature_dimension = 3,
      output_spatial_dimensions = [1, 2]
    >,
    feature_group_count = 1 : i64,
    lhs_dilation = dense<1> : tensor<2xi64>,
    padding = dense<2> : tensor<2x2xi64>,
    rhs_dilation = dense<1> : tensor<2xi64>,
    window_strides = dense<1> : tensor<2xi64>
  } : (tensor<100x26x26x32xf32>, tensor<1x3x3x32xf32>) -> tensor<100x28x28x1xf32>
  %min = stablehlo.constant dense<0.0> : tensor<100x28x28x1xf32>
  %max = stablehlo.constant dense<0xFFFFFFFF> : tensor<100x28x28x1xf32>
  %result = stablehlo.clamp %min, %conv, %max : tensor<100x28x28x1xf32>
  func.return %result : tensor<100x28x28x1xf32>
}
