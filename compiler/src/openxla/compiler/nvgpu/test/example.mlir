// RUN: iree-opt %s --iree-plugin=openxla_nvgpu --pass-pipeline='builtin.module(openxla-nvgpu-convert-hlo-to-cudnn)' | FileCheck %s

!tensor = tensor<1x16x32x8xf32>

// CHECK: cudnn.graph @stablehlo.clamp
// CHECK:   cudnn.pointwise_relu
// CHECK:   cudnn.return

// CHECK: func.func @conv
func.func @conv(%argument: !tensor) -> !tensor {
  %min = stablehlo.constant dense<0.0> : !tensor
  %max = stablehlo.constant dense<0xFFFFFFFF> : !tensor
  // CHECK: cudnn.call @stablehlo.clamp
  %result = stablehlo.clamp %min, %argument, %max : !tensor
  return %result : !tensor
}