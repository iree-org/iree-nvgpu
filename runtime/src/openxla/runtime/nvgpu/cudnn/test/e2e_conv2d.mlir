// RUN: iree-opt %s --iree-plugin=openxla_nvgpu                                \
// RUN:     --openxla-nvgpu-convert-hlo-to-cudnn                               \
// RUN: | iree-compile - --iree-plugin=openxla_nvgpu --iree-input-type=mhlo    \
// RUN:     --iree-hal-target-backends=cuda                                    \
// RUN: | iree-run-module --module=- --device=cuda --function=test_conv        \
// RUN: | FileCheck %s


util.global @x : tensor<100x26x26x32xf32> = dense<1.0> : tensor<100x26x26x32xf32>
util.global @w : tensor<32x3x3x32xf32> = dense<1.0> : tensor<32x3x3x32xf32>

// CHECK: @test_conv
// CHECK: 100x26x26x32xf32
// CHECK: [128 128 128 128 128
// CHECK: [192 192 192 192 192
func.func @test_conv() -> tensor<100x26x26x32xf32> {
  %x = util.global.load @x : tensor<100x26x26x32xf32>
  %w = util.global.load @w : tensor<32x3x3x32xf32>
  %result = "stablehlo.convolution"(%x, %w) {
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
      window_strides = dense<1> : tensor<2xi64> }
      : (tensor<100x26x26x32xf32>, tensor<32x3x3x32xf32>)
      -> tensor<100x26x26x32xf32>
  func.return %result : tensor<100x26x26x32xf32>
}
