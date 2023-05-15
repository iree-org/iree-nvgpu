// RUN: iree-compile %s --iree-plugin=openxla_nvgpu                            \
// RUN:     --iree-input-type=stablehlo --iree-hal-target-backends=cuda        \
// RUN: | iree-run-module --module=- --device=cuda --function=predict          \
// RUN:     --input=1x56x56x64xf32                                             \
// RUN: FileCheck %s


// CHECK: EXEC @predict
// CHECK: 1x56x56x256xf32
// CHECK: [0.0539055 0.0539055 0.0539055

util.global private @"__iree_flow___sm_node317__m.layer-23.gamma" {noinline} = dense<0.025641026> : tensor<64xf32>
util.global private @"__iree_flow___sm_node318__m.layer-23.beta" {noinline} = dense<2.500000e-02> : tensor<64xf32>
util.global private @"__iree_flow___sm_node319__m.layer-23.moving_mean" {noinline} = dense<0.024390243> : tensor<64xf32>
util.global private @"__iree_flow___sm_node320__m.layer-23.moving_variance" {noinline} = dense<0.0238095243> : tensor<64xf32>
util.global private @"__iree_flow___sm_node329__m.layer-25.kernel" {noinline} = dense<0.0232558139> : tensor<1x1x64x256xf32>
util.global private @"__iree_flow___sm_node330__m.layer-25.bias" {noinline} = dense<0.0227272734> : tensor<256xf32>

func.func @predict(%arg0: tensor<1x56x56x64xf32>) -> tensor<1x56x56x256xf32> attributes {iree.module.export, iree.reflection = {abi = "sip", abiv = 1 : i32, sip = "I8!S5!k0_0R3!_0"}} {
  %a = util.global.address @"__iree_flow___sm_node317__m.layer-23.gamma" : !util.ptr<tensor<64xf32>>
  %b = util.global.address @"__iree_flow___sm_node318__m.layer-23.beta" : !util.ptr<tensor<64xf32>>
  %c = util.global.address @"__iree_flow___sm_node319__m.layer-23.moving_mean" : !util.ptr<tensor<64xf32>>
  %d = util.global.address @"__iree_flow___sm_node320__m.layer-23.moving_variance" : !util.ptr<tensor<64xf32>>
  %e = util.global.address @"__iree_flow___sm_node329__m.layer-25.kernel" : !util.ptr<tensor<1x1x64x256xf32>>
  %f = util.global.address @"__iree_flow___sm_node330__m.layer-25.bias" : !util.ptr<tensor<256xf32>>
  %0 = stablehlo.constant dense<0.000000e+00> : tensor<1x56x56x64xf32>
  %1 = util.global.load.indirect %d : !util.ptr<tensor<64xf32>> -> tensor<64xf32>
  %2 = util.global.load.indirect %c : !util.ptr<tensor<64xf32>> -> tensor<64xf32>
  %3 = util.global.load.indirect %b : !util.ptr<tensor<64xf32>> -> tensor<64xf32>
  %4 = util.global.load.indirect %a : !util.ptr<tensor<64xf32>> -> tensor<64xf32>
  %5 = util.global.load.indirect %f : !util.ptr<tensor<256xf32>> -> tensor<256xf32>
  %6 = util.global.load.indirect %e : !util.ptr<tensor<1x1x64x256xf32>> -> tensor<1x1x64x256xf32>
  %7 = "stablehlo.batch_norm_inference"(%arg0, %4, %3, %2, %1) {epsilon = 1.001000e-05 : f32, feature_index = 3 : i64} : (tensor<1x56x56x64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) -> tensor<1x56x56x64xf32>
  %8 = stablehlo.maximum %7, %0 : tensor<1x56x56x64xf32>
  %9 = stablehlo.convolution(%8, %6) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x56x56x64xf32>, tensor<1x1x64x256xf32>) -> tensor<1x56x56x256xf32>
  %10 = stablehlo.broadcast_in_dim %5, dims = [3] : (tensor<256xf32>) -> tensor<1x56x56x256xf32>
  %11 = stablehlo.add %9, %10 : tensor<1x56x56x256xf32>
  return %11 : tensor<1x56x56x256xf32>
}
