// RUN: iree-compile %s --iree-plugin=openxla-cudnn                            \
// RUN:     --iree-input-type=stablehlo --compile-to=vm                        \
// RUN:     --iree-hal-target-backends=cuda                                    \
// RUN: | FileCheck %s

//===-----------------------------------------------------------------------===/
// 1x1 convolution
//===-----------------------------------------------------------------------===/

// CHECK: vm.func private @conv2d_1x1
func.func @conv2d_1x1(
  %x: tensor<8x256x256x32xf32>,
  %w: tensor<1x1x32x32xf32>,
  %b: tensor<8x256x256x32xf32>,
  %c: tensor<32xf32>
) -> tensor<8x256x256x32xf32> {

  %0 = "stablehlo.convolution"(%x, %w) {
    batch_group_count = 1 : i64,
    dimension_numbers = #stablehlo.conv<raw
      input_batch_dimension = 0,
      input_feature_dimension = 3,
      input_spatial_dimensions = [1, 2],
      kernel_input_feature_dimension = 2,
      kernel_output_feature_dimension = 3,
      kernel_spatial_dimensions = [0, 1],
      output_batch_dimension = 0,
      output_feature_dimension = 3,
      output_spatial_dimensions = [1, 2]>,
    feature_group_count = 1 : i64,
    padding = dense<0> : tensor<2x2xi64>,
    rhs_dilation = dense<1> : tensor<2xi64>,
    window_strides = dense<1> : tensor<2xi64>
  } : (tensor<8x256x256x32xf32>, tensor<1x1x32x32xf32>)
    -> tensor<8x256x256x32xf32>

  %1 = stablehlo.add %0, %b : tensor<8x256x256x32xf32>
  %2 = "stablehlo.broadcast_in_dim"(%c)
        { broadcast_dimensions = dense<3> : tensor<1xi64> }
        : (tensor<32xf32>) -> tensor<8x256x256x32xf32>
  %3 = stablehlo.add %1, %2 : tensor<8x256x256x32xf32>

  return %3 : tensor<8x256x256x32xf32>
}

//===-----------------------------------------------------------------------===/
// 3x3 convolution
//===-----------------------------------------------------------------------===/

// CHECK: vm.func private @conv2d_3x3
func.func @conv2d_3x3(
  %x: tensor<8x256x256x32xf32>,
  %w: tensor<3x3x32x32xf32>,
  %b: tensor<8x256x256x32xf32>,
  %c: tensor<32xf32>
) -> tensor<8x256x256x32xf32> {

  %0 = "stablehlo.convolution"(%x, %w) {
    batch_group_count = 1 : i64,
    dimension_numbers = #stablehlo.conv<raw
      input_batch_dimension = 0,
      input_feature_dimension = 3,
      input_spatial_dimensions = [1, 2],
      kernel_input_feature_dimension = 2,
      kernel_output_feature_dimension = 3,
      kernel_spatial_dimensions = [0, 1],
      output_batch_dimension = 0,
      output_feature_dimension = 3,
      output_spatial_dimensions = [1, 2]>,
    feature_group_count = 1 : i64,
    padding = dense<1> : tensor<2x2xi64>,
    rhs_dilation = dense<1> : tensor<2xi64>,
    window_strides = dense<1> : tensor<2xi64>
  } : (tensor<8x256x256x32xf32>, tensor<3x3x32x32xf32>)
    -> tensor<8x256x256x32xf32>

  %1 = stablehlo.add %0, %b : tensor<8x256x256x32xf32>
  %2 = "stablehlo.broadcast_in_dim"(%c)
        { broadcast_dimensions = dense<3> : tensor<1xi64> }
        : (tensor<32xf32>) -> tensor<8x256x256x32xf32>
  %3 = stablehlo.add %1, %2 : tensor<8x256x256x32xf32>

  return %3 : tensor<8x256x256x32xf32>
}
