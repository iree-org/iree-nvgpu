// RUN: iree-opt --iree-plugin=openxla_nvgpu %s | iree-opt --iree-plugin=openxla_nvgpu | FileCheck %s

// CHECK: @foo(%arg0: !cudnn.tensor<?x?x?xf32>) -> !cudnn.tensor<?x?x?xf32>
cudnn.graph @foo(%arg0: !cudnn.tensor<?x?x?xf32>) -> !cudnn.tensor<?x?x?xf32> {
  // CHECK: return %arg0: !cudnn.tensor<?x?x?xf32>
  cudnn.return %arg0: !cudnn.tensor<?x?x?xf32>
}
