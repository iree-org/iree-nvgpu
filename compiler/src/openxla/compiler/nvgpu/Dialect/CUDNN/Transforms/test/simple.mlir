// RUN: iree-opt %s --iree-plugin=openxla_nvgpu %s --pass-pipeline='builtin.module(openxla-nvgpu-convert-cudnn-to-runtime)' | FileCheck %s

// CHECK: @graph
cudnn.graph @graph(%arg0: !cudnn.tensor<1x4x8xf32>)
                     -> !cudnn.tensor<1x4x8xf32> {
  cudnn.return %arg0: !cudnn.tensor<1x4x8xf32>
}
