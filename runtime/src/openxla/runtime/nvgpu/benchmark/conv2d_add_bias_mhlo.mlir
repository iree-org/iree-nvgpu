module @example {

//===-----------------------------------------------------------------------===/
// 1x1 convolution
//===-----------------------------------------------------------------------===/

func.func @conv2d_1x1(
  %x: tensor<8x256x256x32xf32>,
  %w: tensor<1x1x32x32xf32>,
  %b: tensor<8x256x256x32xf32>,
  %c: tensor<32xf32>
) -> tensor<8x256x256x32xf32> {

  %0 = "mhlo.convolution"(%x, %w) {
    batch_group_count = 1 : i64,
    dimension_numbers = #mhlo.conv<raw
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

  %1 = mhlo.add %0, %b : tensor<8x256x256x32xf32>
  %2 = "mhlo.broadcast_in_dim"(%c)
        { broadcast_dimensions = dense<3> : tensor<1xi64> }
        : (tensor<32xf32>) -> tensor<8x256x256x32xf32>
  %3 = mhlo.add %1, %2 : tensor<8x256x256x32xf32>

  return %3 : tensor<8x256x256x32xf32>
}

//===-----------------------------------------------------------------------===/
// 3x3 convolution
//===-----------------------------------------------------------------------===/

func.func @conv2d_3x3(
  %x: tensor<8x256x256x32xf32>,
  %w: tensor<3x3x32x32xf32>,
  %b: tensor<8x256x256x32xf32>,
  %c: tensor<32xf32>
) -> tensor<8x256x256x32xf32> {

  %0 = "mhlo.convolution"(%x, %w) {
    batch_group_count = 1 : i64,
    dimension_numbers = #mhlo.conv<raw
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

  %1 = mhlo.add %0, %b : tensor<8x256x256x32xf32>
  %2 = "mhlo.broadcast_in_dim"(%c)
        { broadcast_dimensions = dense<3> : tensor<1xi64> }
        : (tensor<32xf32>) -> tensor<8x256x256x32xf32>
  %3 = mhlo.add %1, %2 : tensor<8x256x256x32xf32>

  return %3 : tensor<8x256x256x32xf32>
}

}
