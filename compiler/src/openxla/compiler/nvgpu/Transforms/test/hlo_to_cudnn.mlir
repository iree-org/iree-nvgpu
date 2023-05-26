// RUN: iree-opt %s --split-input-file --iree-plugin=openxla-cudnn             \
// RUN:     --openxla-nvgpu-convert-hlo-to-cudnn                               \
// RUN: | FileCheck %s


func.func @test_relu(%argument: tensor<1x16x32x8xf32>)
    -> tensor<1x16x32x8xf32> {
  %min = stablehlo.constant dense<0.0> : tensor<1x16x32x8xf32>
  %max = stablehlo.constant dense<0xFFFFFFFF> : tensor<1x16x32x8xf32>
  %result = stablehlo.clamp %min, %argument, %max : tensor<1x16x32x8xf32>
  return %result : tensor<1x16x32x8xf32>
}

// CHECK:      util.global public @stablehlo.clamp.handle
// CHECK:      util.initializer
// CHECK:        %[[DEVICE:.*]] = hal.ex.shared_device
// CHECK:        %[[HANDLE:.*]] = cudnn.handle(%[[DEVICE]])
// CHECK:        util.global.store %[[HANDLE]], @stablehlo.clamp.handle
// CHECK:        util.initializer.return

// CHECK:      cudnn.graph @stablehlo.clamp(%[[ARG0:.*]]: !cudnn.tensor<1x16x32x8xf32, NCHW>)
// CHECK-SAME:     -> !cudnn.tensor<1x16x32x8xf32, NCHW>
// CHECK:        %[[RELU:.*]] = cudnn.relu(%[[ARG0]])
// CHECK-SAME:       type = f32 lower_clip = 0.000000e+00
// CHECK:        cudnn.return %[[RELU]]

// CHECK:      @test_relu(%[[ARG0]]: tensor<1x16x32x8xf32>)
// CHECK:        %[[STABLEHLO:.*]].clamp.handle = util.global.load @stablehlo.clamp.handle
// CHECK:        %[[CALL:.*]] = cudnn.call handle(%[[STABLEHLO]].clamp.handle) @stablehlo.clamp(%[[ARG0]])
// CHECK:        return %[[CALL]]

// -----

func.func @test_conv(%input : tensor<100x32x26x26xf32>,
    %filter : tensor<1x32x3x3xf32>) -> tensor<100x1x28x28xf32> {
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
        output_spatial_dimensions = [2, 3]>,
      feature_group_count = 1 : i64,
      lhs_dilation = dense<1> : tensor<2xi64>,
      padding = dense<2> : tensor<2x2xi64>,
      rhs_dilation = dense<1> : tensor<2xi64>,
      window_strides = dense<1> : tensor<2xi64>}
      : (tensor<100x32x26x26xf32>, tensor<1x32x3x3xf32>)
      -> tensor<100x1x28x28xf32>
  func.return %result : tensor<100x1x28x28xf32>
}

// CHECK:        util.global public @stablehlo.convolution.handle
// CHECK:        util.initializer
// CHECK:          %[[DEVICE_0:.*]] = hal.ex.shared_device
// CHECK:          %[[HANDLE_0:.*]] = cudnn.handle(%[[DEVICE_0]])
// CHECK:          util.global.store %[[HANDLE_0]], @stablehlo.convolution.handle
// CHECK:          util.initializer.return

// CHECK:        cudnn.graph @stablehlo.convolution(%[[ARG0_0:.*]]: !cudnn.tensor<100x32x26x26xf32, NCHW>, %[[ARG1:.*]]: !cudnn.tensor<1x32x3x3xf32, NCHW>)
// CHECK-SAME:       -> !cudnn.tensor<100x1x28x28xf32, NCHW>
// CHECK:          %[[CONVOLUTION:.*]] = cudnn.convolution(%[[ARG0_0]], %[[ARG1]])
// CHECK-SAME:         alpha = 1.000000e+00
// CHECK-SAME:         beta = 0.000000e+00
// CHECK-SAME:         spatial_dim_count = 2
// CHECK-SAME:         spatial_stride = [1, 1]
// CHECK-SAME:         pre_padding = [2, 2]
// CHECK-SAME:         post_padding = [2, 2]
// CHECK-SAME:         dilation = [1, 1]
// CHECK:          cudnn.return %[[CONVOLUTION]]

// CHECK:        @test_conv(%[[ARG0_0]]: tensor<100x32x26x26xf32>, %[[ARG1]]: tensor<1x32x3x3xf32>)
// CHECK:          %[[STABLEHLO_0:.*]].convolution.handle = util.global.load @stablehlo.convolution.handle
// CHECK:          %[[CALL_0:.*]] = cudnn.call handle(%[[STABLEHLO_0]].convolution.handle) @stablehlo.convolution(%[[ARG0_0]], %[[ARG1]])
// CHECK:          return %[[CALL_0]]

// -----

func.func @test_graph(
  %input : tensor<100x26x26x32xf32>,
  %filter : tensor<1x3x3x32xf32>
) -> tensor<100x28x28x1xf32> {
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
        output_spatial_dimensions = [1, 2]>,
      feature_group_count = 1 : i64,
      lhs_dilation = dense<1> : tensor<2xi64>,
      padding = dense<2> : tensor<2x2xi64>,
      rhs_dilation = dense<1> : tensor<2xi64>,
      window_strides = dense<1> : tensor<2xi64>}
      : (tensor<100x26x26x32xf32>, tensor<1x3x3x32xf32>)
      -> tensor<100x28x28x1xf32>
  %min = stablehlo.constant dense<0.0> : tensor<100x28x28x1xf32>
  %max = stablehlo.constant dense<0xFFFFFFFF> : tensor<100x28x28x1xf32>
  %result = stablehlo.clamp %min, %conv, %max : tensor<100x28x28x1xf32>
  func.return %result : tensor<100x28x28x1xf32>
}

// CHECK:        util.global public @stablehlo.clamp.handle
// CHECK:        util.initializer
// CHECK:          %[[DEVICE_1:.*]] = hal.ex.shared_device
// CHECK:          %[[HANDLE_1:.*]] = cudnn.handle(%[[DEVICE_1]])
// CHECK:          util.global.store %[[HANDLE_1]], @stablehlo.clamp.handle
// CHECK:          util.initializer.return

// CHECK:        cudnn.graph @stablehlo.clamp(%[[ARG0_1:.*]]: !cudnn.tensor<100x32x26x26xf32, NHWC>, %[[ARG1_0:.*]]: !cudnn.tensor<1x32x3x3xf32, NHWC>)
// CHECK-SAME:       -> !cudnn.tensor<100x1x28x28xf32, NHWC>
// CHECK:          %[[CONVOLUTION_0:.*]] = cudnn.convolution(%[[ARG0_1]], %[[ARG1_0]])
// CHECK-SAME:         alpha = 1.000000e+00
// CHECK-SAME:         beta = 0.000000e+00
// CHECK-SAME:         spatial_dim_count = 2
// CHECK-SAME:         spatial_stride = [1, 1]
// CHECK-SAME:         pre_padding = [2, 2]
// CHECK-SAME:         post_padding = [2, 2]
// CHECK-SAME:         dilation = [1, 1]
// CHECK:          %[[RELU_0:.*]] = cudnn.relu(%[[CONVOLUTION_0]])
// CHECK-SAME:         type = f32
// CHECK-SAME:         lower_clip = 0.000000e+00
// CHECK:          cudnn.return %[[RELU_0]]

// CHECK:        @test_graph(%[[ARG0_1]]: tensor<100x26x26x32xf32>, %[[ARG1_0]]: tensor<1x3x3x32xf32>)
// CHECK:          %[[STABLEHLO_1:.*]].clamp.handle = util.global.load @stablehlo.clamp.handle
// CHECK:          %[[CALL_1:.*]] = cudnn.call handle(%[[STABLEHLO_1]].clamp.handle) @stablehlo.clamp(%[[ARG0_1]], %[[ARG1_0]])
// CHECK:          return %[[CALL_1]]

// -----

func.func @multi_conv(%input : tensor<100x26x26x32xf32>,
    %filter : tensor<32x3x3x32xf32>) -> tensor<100x26x26x32xf32> {
  %conv_0 = "stablehlo.convolution"(%input, %filter) {
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
        output_spatial_dimensions = [1, 2]>,
      feature_group_count = 1 : i64,
      lhs_dilation = dense<1> : tensor<2xi64>,
      padding = dense<1> : tensor<2x2xi64>,
      rhs_dilation = dense<1> : tensor<2xi64>,
      window_strides = dense<1> : tensor<2xi64>}
      : (tensor<100x26x26x32xf32>, tensor<32x3x3x32xf32>)
      -> tensor<100x26x26x32xf32>
  %conv_1 = "stablehlo.convolution"(%conv_0, %filter) {
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
        output_spatial_dimensions = [1, 2]>,
      feature_group_count = 1 : i64,
      lhs_dilation = dense<1> : tensor<2xi64>,
      padding = dense<1> : tensor<2x2xi64>,
      rhs_dilation = dense<1> : tensor<2xi64>,
      window_strides = dense<1> : tensor<2xi64>}
      : (tensor<100x26x26x32xf32>, tensor<32x3x3x32xf32>)
      -> tensor<100x26x26x32xf32>
  func.return %conv_1 : tensor<100x26x26x32xf32>
}

// CHECK:      util.global public @stablehlo.convolution.handle
// CHECK:      util.initializer
// CHECK:        %[[DEVICE:.*]] = hal.ex.shared_device
// CHECK:        %[[HANDLE:.*]] = cudnn.handle(%[[DEVICE]])
// CHECK:        util.global.store %[[HANDLE]], @stablehlo.convolution.handle
// CHECK:        util.initializer.return

// CHECK:      cudnn.graph @stablehlo.convolution(%[[ARG0:.*]]: !cudnn.tensor<100x32x26x26xf32, NHWC>, %[[ARG1:.*]]: !cudnn.tensor<32x32x3x3xf32, NHWC>, %[[ARG2:.*]]: !cudnn.tensor<32x32x3x3xf32, NHWC>)
// CHECK-SAME:     -> !cudnn.tensor<100x32x26x26xf32, NHWC>
// CHECK:        %[[CONVOLUTION:.*]] = cudnn.convolution(%[[ARG0]], %[[ARG2]])
// CHECK-SAME:       alpha = 1.000000e+00
// CHECK-SAME:       beta = 0.000000e+00
// CHECK-SAME:       spatial_dim_count = 2
// CHECK-SAME:       spatial_stride = [1, 1]
// CHECK-SAME:       pre_padding = [1, 1]
// CHECK-SAME:       post_padding = [1, 1]
// CHECK-SAME:       dilation = [1, 1]
// CHECK:        %[[CONVOLUTION_0:.*]] = cudnn.convolution(%[[CONVOLUTION]], %[[ARG2]])
// CHECK-SAME:       alpha = 1.000000e+00
// CHECK-SAME:       beta = 0.000000e+00
// CHECK-SAME:       spatial_dim_count = 2
// CHECK-SAME:       spatial_stride = [1, 1]
// CHECK-SAME:       pre_padding = [1, 1]
// CHECK-SAME:       post_padding = [1, 1]
// CHECK-SAME:       dilation = [1, 1]
// CHECK:        cudnn.return %[[CONVOLUTION_0]]

// CHECK:      @multi_conv(%[[ARG0]]: tensor<100x26x26x32xf32>, %[[ARG1]]: tensor<32x3x3x32xf32>)
// CHECK:        %[[STABLEHLO:.*]].convolution.handle = util.global.load @stablehlo.convolution.handle
// CHECK:        %[[CALL:.*]] = cudnn.call handle(%[[STABLEHLO]].convolution.handle) @stablehlo.convolution(%[[ARG0]], %[[ARG1]], %[[ARG1]])
// CHECK:        return %[[CALL]]
