// RUN: iree-compile %s --iree-plugin=openxla-cudnn                            \
// RUN:     --iree-input-type=stablehlo --iree-hal-target-backends=cuda        \
// RUN:     --mlir-print-ir-after=openxla-nvgpu-convert-cudnn-to-runtime       \
// RUN:     2> %t-ir                                                           \
// RUN: | iree-run-module --module=- --device=cuda --function=predict          \
// RUN:     --input=1x56x56x64xf32                                             \
// RUN: FileCheck %s

// RUN: cat %t-ir                                                              \
// RUN: | FileCheck %s --check-prefix=CHECK-IR


// batch_norm_inference
//     = offset + scale * ((operand - mean) / sqrt(variance + eps))
//     = 0.025 + 0.0256410 * ((0.0 - 0.0243902) / sqrt(0.02380952 + 1.001e-05))
//     = 0.02094785578
// result
//     = convolution_i64(max(batch_norm_inference, 0.0), kernel) + bias
//     = 64 * max(0.02094785578, 0.0) * 0.0232558139 + 0.0227272734
//     = 0.05390547727

// CHECK: EXEC @predict
// CHECK: 1x56x56x256xf32
// CHECK: [0.0539055 0.0539055 0.0539055

util.global private @"__iree_flow___sm_node317__m.layer-23.gamma"
    {noinline} = dense<0.025641026> : tensor<64xf32>
util.global private @"__iree_flow___sm_node318__m.layer-23.beta"
    {noinline} = dense<2.500000e-02> : tensor<64xf32>
util.global private @"__iree_flow___sm_node319__m.layer-23.moving_mean"
    {noinline} = dense<0.024390243> : tensor<64xf32>
util.global private @"__iree_flow___sm_node320__m.layer-23.moving_variance"
    {noinline} = dense<0.0238095243> : tensor<64xf32>
util.global private @"__iree_flow___sm_node329__m.layer-25.kernel"
    {noinline} = dense<0.0232558139> : tensor<1x1x64x256xf32>
util.global private @"__iree_flow___sm_node330__m.layer-25.bias"
    {noinline} = dense<0.0227272734> : tensor<256xf32>

func.func @predict(%arg0: tensor<1x56x56x64xf32>) -> tensor<1x56x56x256xf32>
    attributes {iree.module.export, iree.reflection = {abi = "sip",
    abiv = 1 : i32, sip = "I8!S5!k0_0R3!_0"}} {
  %gamma_ptr = util.global.address @"__iree_flow___sm_node317__m.layer-23.gamma"
      : !util.ptr<tensor<64xf32>>
  %beta_ptr = util.global.address @"__iree_flow___sm_node318__m.layer-23.beta"
      : !util.ptr<tensor<64xf32>>
  %mean_ptr = util.global.address
      @"__iree_flow___sm_node319__m.layer-23.moving_mean"
      : !util.ptr<tensor<64xf32>>
  %variance_ptr = util.global.address
      @"__iree_flow___sm_node320__m.layer-23.moving_variance"
      : !util.ptr<tensor<64xf32>>
  %kernel_ptr = util.global.address
      @"__iree_flow___sm_node329__m.layer-25.kernel"
      : !util.ptr<tensor<1x1x64x256xf32>>
  %bias_ptr = util.global.address @"__iree_flow___sm_node330__m.layer-25.bias"
  : !util.ptr<tensor<256xf32>>
  %zero = stablehlo.constant dense<0.000000e+00> : tensor<1x56x56x64xf32>
  %variance = util.global.load.indirect %variance_ptr
      : !util.ptr<tensor<64xf32>> -> tensor<64xf32>
  %mean = util.global.load.indirect %mean_ptr
      : !util.ptr<tensor<64xf32>> -> tensor<64xf32>
  %beta = util.global.load.indirect %beta_ptr
      : !util.ptr<tensor<64xf32>> -> tensor<64xf32>
  %gamma = util.global.load.indirect %gamma_ptr
      : !util.ptr<tensor<64xf32>> -> tensor<64xf32>
  %bias = util.global.load.indirect %bias_ptr
      : !util.ptr<tensor<256xf32>> -> tensor<256xf32>
  %kernel = util.global.load.indirect %kernel_ptr
      : !util.ptr<tensor<1x1x64x256xf32>> -> tensor<1x1x64x256xf32>
  %batch_norm_inference = "stablehlo.batch_norm_inference"(%arg0, %gamma, %beta,
      %mean, %variance) {epsilon = 1.001000e-05 : f32, feature_index = 3 : i64}
      : (tensor<1x56x56x64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>,
      tensor<64xf32>) -> tensor<1x56x56x64xf32>
  %max = stablehlo.maximum %batch_norm_inference, %zero : tensor<1x56x56x64xf32>
  %conv = stablehlo.convolution(%max, %kernel)
      dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {
      batch_group_count = 1 : i64, feature_group_count = 1 : i64 }
      : (tensor<1x56x56x64xf32>, tensor<1x1x64x256xf32>)
      -> tensor<1x56x56x256xf32>
  %bcast_bias = stablehlo.broadcast_in_dim %bias, dims = [3] : (tensor<256xf32>)
      -> tensor<1x56x56x256xf32>
  %result = stablehlo.add %conv, %bcast_bias : tensor<1x56x56x256xf32>
  return %result : tensor<1x56x56x256xf32>
}


// Ensure that we actually lower to Cudnn.
// CHECK-IR: IR Dump After ConvertCudnnToRuntime (openxla-nvgpu-convert-cudnn-to-runtime)

// CHECK-IR: func.func @stablehlo.convolution.builder
// CHECK-IR:   call @cudnn.tensor.create.4d.nhwc
// CHECK-IR:   call @cudnn.tensor.create.4d.nhwc
// CHECK-IR:   call @cudnn.convolution.2d
// CHECK-IR:   call @cudnn.operation_graph.create

// CHECK-IR: func.func private @_predict
// CHECK-IR:   call @cudnn.handle
// CHECK-IR:   call @stablehlo.convolution.builder
// CHECK-IR:   call @cudnn.executable.create
// CHECK-IR:   call @cudnn.execute.2
