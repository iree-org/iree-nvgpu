// RUN: iree-opt %s --iree-plugin=openxla_nvgpu --split-input-file \
// RUN:   --pass-pipeline='builtin.module(openxla-nvgpu-convert-cudnn-to-runtime)' \
// RUN:   | FileCheck %s

cudnn.graph @graph(%arg0: !cudnn.tensor<1x4x8xf32>)
                     -> !cudnn.tensor<1x4x8xf32> {
  cudnn.return %arg0: !cudnn.tensor<1x4x8xf32>
}

// CHECK: func.func @graph.builder() -> !cudnn.operation_graph {
// CHECK:   %[[DT:.*]] = arith.constant 0 : i64
// CHECK:   %[[D0:.*]] = arith.constant 1 : i64
// CHECK:   %[[D1:.*]] = arith.constant 4 : i64
// CHECK:   %[[D2:.*]] = arith.constant 8 : i64
// CHECK:   %[[ARG:.*]] = call @cudnn.tensor.create.3d(%[[DT]], %[[D0]], %[[D1]], %[[D2]])
// CHECK:   %[[GRAPH:.*]] = call @cudnn.operation_graph.create(%[[ARG0]])
// CHECK:   return %[[GRAPH]] : !cudnn.operation_graph
// CHECK: }

// CHECK: @cudnn.tensor.create.3d(i64, i64, i64, i64) -> !cudnn.tensor
// CHECK: @cudnn.operation_graph.create(!cudnn.tensor) -> !cudnn.operation_graph
