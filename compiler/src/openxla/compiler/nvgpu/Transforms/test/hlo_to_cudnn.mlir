// RUN: iree-opt %s --split-input-file --iree-plugin=openxla-cudnn             \
// RUN:     --openxla-nvgpu-convert-hlo-to-cudnn                               \
// RUN: | FileCheck %s

func.func @conv(%input : tensor<100x32x26x26xf32>,
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

// CHECK:      util.global public @cudnn.shared.handle
// CHECK:      util.initializer
// CHECK:        %[[DEVICE:.*]] = hal.ex.shared_device
// CHECK:        %[[HANDLE:.*]] = cudnn.handle(%[[DEVICE]])
// CHECK:        util.global.store %[[HANDLE]], @cudnn.shared.handle
// CHECK:        util.initializer.return

// CHECK:      cudnn.graph @stablehlo.convolution(
// CHECK-SAME:     %[[ARG0:.*]]: !cudnn.tensor<100x32x26x26xf32, NHWC>,
// CHECK-SAME:     %[[ARG1:.*]]: !cudnn.tensor<1x32x3x3xf32, KHWC>)
// CHECK-SAME:     -> !cudnn.tensor<100x1x28x28xf32, NHWC>
// CHECK:        %[[CONVOLUTION:.*]] = cudnn.convolution(%[[ARG0]],
// CHECK-SAME:       %[[ARG1]]) alpha = 1.000000e+00 beta = 0.000000e+00
// CHECK-SAME:       spatial_dim_count = 2 spatial_stride = [1, 1]
// CHECK-SAME:       pre_padding = [2, 2] post_padding = [2, 2]
// CHECK-SAME:       dilation = [1, 1]
// CHECK:        cudnn.return %[[CONVOLUTION]]

// CHECK:      @conv(%[[ARG0]]: tensor<100x32x26x26xf32>,
// CHECK-SAME:     %[[ARG1]]: tensor<1x32x3x3xf32>)
// CHECK:        %[[CUDNN:.*]].shared.handle = util.global.load
// CHECK-SAME:       @cudnn.shared.handle
// CHECK:        %[[TRANSPOSE:.*]] = stablehlo.transpose %[[ARG0]],
// CHECK-SAME:       dims = [0, 2, 3, 1]
// CHECK:        %[[TRANSPOSE_0:.*]] = stablehlo.transpose %[[ARG1]],
// CHECK-SAME:       dims = [0, 2, 3, 1]
// CHECK:        %[[CALL:.*]] = cudnn.call handle(%[[CUDNN]].shared.handle)
// CHECK-SAME:       @stablehlo.convolution(%[[TRANSPOSE]], %[[TRANSPOSE_0]])
// CHECK:        %[[TRANSPOSE_1:.*]] = stablehlo.transpose %[[CALL]],
// CHECK-SAME:       dims = [0, 3, 1, 2]
// CHECK:        return %[[TRANSPOSE_1]]

// -----

func.func @conv_relu(%input : tensor<100x26x26x32xf32>,
    %filter : tensor<1x3x3x32xf32>) -> tensor<100x28x28x1xf32> {
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

// CHECK:      util.global public @cudnn.shared.handle
// CHECK:      util.initializer
// CHECK:        %[[DEVICE_0:.*]] = hal.ex.shared_device
// CHECK:        %[[HANDLE_0:.*]] = cudnn.handle(%[[DEVICE_0]])
// CHECK:        util.global.store %[[HANDLE_0]], @cudnn.shared.handle
// CHECK:        util.initializer.return

// CHECK:      cudnn.graph @stablehlo.convolution(
// CHECK-SAME:     %[[ARG0_0:.*]]: !cudnn.tensor<100x32x26x26xf32, NHWC>,
// CHECK-SAME:     %[[ARG1_0:.*]]: !cudnn.tensor<1x32x3x3xf32, KHWC>)
// CHECK-SAME:     -> !cudnn.tensor<100x1x28x28xf32, NHWC>
// CHECK:        %[[CONVOLUTION_0:.*]] = cudnn.convolution(%[[ARG0_0]],
// CHECK-SAME:       %[[ARG1_0]]) alpha = 1.000000e+00 beta = 0.000000e+00
// CHECK-SAME:       spatial_dim_count = 2 spatial_stride = [1, 1]
// CHECK-SAME:       pre_padding = [2, 2] post_padding = [2, 2]
// CHECK-SAME:       dilation = [1, 1]
// CHECK:        %[[RELU:.*]] = cudnn.relu(%[[CONVOLUTION_0]]) type = f32
// CHECK-SAME:       lower_clip = 0.000000e+00
// CHECK:        cudnn.return %[[RELU]]

// CHECK:      @conv_relu(%[[ARG0_0]]: tensor<100x26x26x32xf32>,
// CHECK-SAME:     %[[ARG1_0]]: tensor<1x3x3x32xf32>)
// CHECK:        %[[CUDNN_0:.*]].shared.handle = util.global.load
// CHECK-SAME:       @cudnn.shared.handle
// CHECK:        %[[CALL_0:.*]] = cudnn.call handle(%[[CUDNN_0]].shared.handle)
// CHECK-SAME:       @stablehlo.convolution(%[[ARG0_0]], %[[ARG1_0]])
// CHECK:        return %[[CALL_0]]

// -----

func.func @relu_conv(%input : tensor<100x32x26x26xf32>,
    %filter : tensor<1x32x3x3xf32>) -> tensor<100x1x28x28xf32> {
  %zero = stablehlo.constant dense<0.0> : tensor<100x32x26x26xf32>
  %max = stablehlo.constant dense<0xFFFFFFFF> : tensor<100x32x26x26xf32>
  %relu = stablehlo.clamp %zero, %input, %max : tensor<100x32x26x26xf32>
  %result = "stablehlo.convolution"(%relu, %filter) {
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

// CHECK:      util.global public @cudnn.shared.handle
// CHECK:      util.initializer
// CHECK:        %[[DEVICE_1:.*]] = hal.ex.shared_device
// CHECK:        %[[HANDLE_1:.*]] = cudnn.handle(%[[DEVICE_1]])
// CHECK:        util.global.store %[[HANDLE_1]], @cudnn.shared.handle
// CHECK:        util.initializer.return

// CHECK:      cudnn.graph @stablehlo.convolution(
// CHECK-SAME:     %[[ARG0_1:.*]]: !cudnn.tensor<100x32x26x26xf32, NHWC>,
// CHECK-SAME:     %[[ARG1_1:.*]]: !cudnn.tensor<1x32x3x3xf32, KHWC>)
// CHECK-SAME:     -> !cudnn.tensor<100x1x28x28xf32, NHWC>
// CHECK:        %[[RELU_0:.*]] = cudnn.relu(%[[ARG0_1]]) type = f32
// CHECK-SAME:       lower_clip = 0.000000e+00
// CHECK:        %[[CONVOLUTION_1:.*]] = cudnn.convolution(%[[RELU_0]],
// CHECK-SAME:       %[[ARG1_1]]) alpha = 1.000000e+00 beta = 0.000000e+00
// CHECK-SAME:       spatial_dim_count = 2 spatial_stride = [1, 1]
// CHECK-SAME:       pre_padding = [2, 2] post_padding = [2, 2]
// CHECK-SAME:       dilation = [1, 1]
// CHECK:        cudnn.return %[[CONVOLUTION_1]]

// CHECK:      @relu_conv(%[[ARG0_1]]: tensor<100x32x26x26xf32>,
// CHECK-SAME:     %[[ARG1_1]]: tensor<1x32x3x3xf32>)
// CHECK:        %[[CUDNN_1:.*]].shared.handle = util.global.load
// CHECK-SAME:       @cudnn.shared.handle
// CHECK:        %[[TRANSPOSE_2:.*]] = stablehlo.transpose %[[ARG0_1]],
// CHECK-SAME:       dims = [0, 2, 3, 1]
// CHECK:        %[[TRANSPOSE_3:.*]] = stablehlo.transpose %[[ARG1_1]],
// CHECK-SAME:       dims = [0, 2, 3, 1]
// CHECK:        %[[CALL_1:.*]] = cudnn.call handle(%[[CUDNN_1]].shared.handle)
// CHECK-SAME:       @stablehlo.convolution(%[[TRANSPOSE_2]], %[[TRANSPOSE_3]])
// CHECK:        %[[TRANSPOSE_4:.*]] = stablehlo.transpose %[[CALL_1]],
// CHECK-SAME:       dims = [0, 3, 1, 2]
// CHECK:        return %[[TRANSPOSE_4]]

// -----

func.func @conv_conv(%input : tensor<100x26x26x32xf32>,
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

// CHECK:      util.global public @cudnn.shared.handle
// CHECK:      util.initializer
// CHECK:        %[[DEVICE_2:.*]] = hal.ex.shared_device
// CHECK:        %[[HANDLE_2:.*]] = cudnn.handle(%[[DEVICE_2]])
// CHECK:        util.global.store %[[HANDLE_2]], @cudnn.shared.handle
// CHECK:        util.initializer.return

// CHECK:      cudnn.graph @stablehlo.convolution(
// CHECK-SAME:     %[[ARG0_2:.*]]: !cudnn.tensor<100x32x26x26xf32, NHWC>,
// CHECK-SAME:     %[[ARG1_2:.*]]: !cudnn.tensor<32x32x3x3xf32, KHWC>,
// CHECK-SAME:     %[[ARG2:.*]]: !cudnn.tensor<32x32x3x3xf32, KHWC>)
// CHECK-SAME:     -> !cudnn.tensor<100x32x26x26xf32, NHWC>
// CHECK:        %[[CONVOLUTION_2:.*]] = cudnn.convolution(%[[ARG0_2]],
// CHECK-SAME:       %[[ARG1_2]]) alpha = 1.000000e+00 beta = 0.000000e+00
// CHECK-SAME:       spatial_dim_count = 2 spatial_stride = [1, 1]
// CHECK-SAME:       pre_padding = [1, 1] post_padding = [1, 1]
// CHECK-SAME:       dilation = [1, 1]
// CHECK:        %[[CONVOLUTION_3:.*]] = cudnn.convolution(%[[CONVOLUTION_2]],
// CHECK-SAME:       %[[ARG2]]) alpha = 1.000000e+00 beta = 0.000000e+00
// CHECK-SAME:       spatial_dim_count = 2 spatial_stride = [1, 1]
// CHECK-SAME:       pre_padding = [1, 1] post_padding = [1, 1]
// CHECK-SAME:       dilation = [1, 1]
// CHECK:        cudnn.return %[[CONVOLUTION_3]]

// CHECK:      @conv_conv(%[[ARG0_2]]: tensor<100x26x26x32xf32>,
// CHECK-SAME:     %[[ARG1_2]]: tensor<32x3x3x32xf32>)
// CHECK:        %[[CUDNN_2:.*]].shared.handle = util.global.load
// CHECK-SAME:       @cudnn.shared.handle
// CHECK:        %[[CALL_2:.*]] = cudnn.call handle(%[[CUDNN_2]].shared.handle)
// CHECK-SAME:       @stablehlo.convolution(%[[ARG0_2]], %[[ARG1_2]],
// CHECK-SAME:       %[[ARG1_2]])
// CHECK:        return %[[CALL_2]]

// -----

func.func @conv_with_default_precision(%arg0: tensor<1x14x14x512xf32>,
    %arg1: tensor<3x3x512x512xf32>, %arg2: tensor<1x7x7x512xf32>)
    -> tensor<1x7x7x512xf32> {
  %0 = stablehlo.convolution(%arg0, %arg1)
      dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f],
      window = {stride = [2, 2], pad = [[1, 1], [1, 1]],
      lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [0, 0]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64,
      precision_config = [#stablehlo<precision DEFAULT>,
      #stablehlo<precision DEFAULT>]}
      : (tensor<1x14x14x512xf32>, tensor<3x3x512x512xf32>)
      -> tensor<1x7x7x512xf32>
  %1 = stablehlo.subtract %0, %arg2 : tensor<1x7x7x512xf32>
  return %1 : tensor<1x7x7x512xf32>
}

// CHECK:      util.global public @cudnn.shared.handle
// CHECK:      util.initializer
// CHECK:        %[[DEVICE:.*]] = hal.ex.shared_device
// CHECK:        %[[HANDLE:.*]] = cudnn.handle(%[[DEVICE]])
// CHECK:        util.global.store %[[HANDLE]], @cudnn.shared.handle
// CHECK:        util.initializer.return
// CHECK:      cudnn.graph @stablehlo.convolution(
// CHECK-SAME:     %[[ARG0:.*]]: !cudnn.tensor<1x512x14x14xf32, NHWC>,
// CHECK-SAME:     %[[ARG1:.*]]: !cudnn.tensor<512x512x3x3xf32, KHWC>)
// CHECK-SAME:     -> !cudnn.tensor<1x512x7x7xf32, NHWC>
// CHECK:        %[[CONVOLUTION:.*]] = cudnn.convolution(%[[ARG0]], %[[ARG1]])
// CHECK-SAME:       alpha = 1.000000e+00 beta = 0.000000e+00
// CHECK-SAME:       spatial_dim_count = 2 spatial_stride = [2, 2]
// CHECK-SAME:       pre_padding = [1, 1] post_padding = [1, 1]
// CHECK-SAME:       dilation = [1, 1]
// CHECK:        cudnn.return %[[CONVOLUTION]]
// CHECK:      @conv_with_default_precision(%[[ARG0]]: tensor<1x14x14x512xf32>,
// CHECK-SAME:     %[[ARG1]]: tensor<3x3x512x512xf32>,
// CHECK-SAME:     %[[ARG2:.*]]: tensor<1x7x7x512xf32>)
// CHECK:        %[[CUDNN:.*]].shared.handle = util.global.load
// CHECK-SAME:       @cudnn.shared.handle
// CHECK:        %[[TRANSPOSE:.*]] = stablehlo.transpose %[[ARG1]],
// CHECK-SAME:       dims = [3, 0, 1, 2]
// CHECK:        %[[CALL:.*]] = cudnn.call handle(%[[CUDNN]].shared.handle)
// CHECK-SAME:       @stablehlo.convolution(%[[ARG0]], %[[TRANSPOSE]])
// CHECK:        %[[SUBTRACT:.*]] = stablehlo.subtract %[[CALL]], %[[ARG2]]
// CHECK:        return %[[SUBTRACT]]
