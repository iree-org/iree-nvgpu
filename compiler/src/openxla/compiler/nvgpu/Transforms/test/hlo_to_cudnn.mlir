// RUN: iree-opt %s --iree-plugin=openxla_nvgpu                                \
// RUN:     --openxla-nvgpu-convert-hlo-to-cudnn                               \
// RUN: | FileCheck %s

!tensor = tensor<1x16x32x8xf32>

// CHECK: cudnn.graph @stablehlo.clamp
// CHECK:   cudnn.relu
// CHECK:   cudnn.return

// CHECK: func.func @test_relu
func.func @test_relu(%argument: !tensor) -> !tensor {
  %min = stablehlo.constant dense<0.0> : !tensor
  %max = stablehlo.constant dense<0xFFFFFFFF> : !tensor
  // CHECK: cudnn.call handle(%{{.*}}) @stablehlo.clamp
  %result = stablehlo.clamp %min, %argument, %max : !tensor
  return %result : !tensor
}

// CHECK: cudnn.graph @stablehlo.convolution
// CHECK:   cudnn.convolution
// CHECK:   cudnn.return

// CHECK: func.func @test_conv
func.func @test_conv(
    %arg0 : tensor<100x32x26x26xf32>,
    %arg1 : tensor<1x32x3x3xf32>
) -> tensor<100x1x28x28xf32> {
  // CHECK: cudnn.call handle(%{{.*}}) @stablehlo.convolution
  %result = "stablehlo.convolution"(%arg0, %arg1) {
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