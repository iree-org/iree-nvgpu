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
// CHECK:   %[[GRAPH:.*]] = call @cudnn.operation_graph.create(%[[ARG]])
// CHECK:   return %[[GRAPH]] : !cudnn.operation_graph
// CHECK: }

// CHECK: @cudnn.tensor.create.3d(i64, i64, i64, i64) -> !cudnn.tensor
// CHECK: @cudnn.operation_graph.create(!cudnn.tensor) -> !cudnn.operation_graph

// -----

cudnn.graph @graph(%arg0: !cudnn.tensor<1x4x8xi32>)
                       -> !cudnn.tensor<1x4x8xi32> {
  cudnn.return %arg0: !cudnn.tensor<1x4x8xi32>
}

// CHECK: func.func @graph.builder() -> !cudnn.operation_graph {
// CHECK:   %[[DT:.*]] = arith.constant 4 : i64
// CHECK:   call @cudnn.tensor.create.3d(%[[DT]],
// CHECK: }

// -----

cudnn.graph @graph(%arg0: !cudnn.tensor<1x4x4x32xi32, NHWC>)
                       -> !cudnn.tensor<1x4x4x32xi32, NHWC> {
  cudnn.return %arg0: !cudnn.tensor<1x4x4x32xi32, NHWC>
}

// CHECK: func.func @graph.builder() -> !cudnn.operation_graph {
// CHECK:   call @cudnn.tensor.create.4d.nhwc
// CHECK: }

// -----

cudnn.graph @add(
  %x: !cudnn.tensor<8x32x4x4xf32, NHWC>,
  %b: !cudnn.tensor<8x32x4x4xf32, NHWC>
) -> !cudnn.tensor<8x32x4x4xf32, NHWC> {
  %0 = cudnn.add(%x, %b) alpha=1.0 alpha2=0.5
    : (!cudnn.tensor<8x32x4x4xf32, NHWC>, !cudnn.tensor<8x32x4x4xf32, NHWC>)
    -> !cudnn.tensor<8x32x4x4xf32, NHWC>
  cudnn.return %0: !cudnn.tensor<8x32x4x4xf32, NHWC>
}

// CHECK: func.func @add.builder() -> !cudnn.operation_graph {
// CHECK:   %[[X:.*]] = call @cudnn.tensor.create.4d.nhwc
// CHECK:   %[[B:.*]] = call @cudnn.tensor.create.4d.nhwc
// CHECK:   %[[ALPHA:.*]] = arith.constant 1.000000e+00 : f32
// CHECK:   %[[ALPHA2:.*]] = arith.constant 5.000000e-01 : f32
// CHECK:   %[[VIRTUAL:.*]] = arith.constant 0 : i32
// CHECK:   %[[Y:.*]] = call @cudnn.add(%[[X]], %[[ALPHA]], %[[B]],
// CHECK:                               %[[ALPHA2]], %[[VIRTUAL]])
// CHECK:   %[[GRAPH:.*]] = call @cudnn.operation_graph.create(%[[Y]])
// CHECK:   return %[[GRAPH]] : !cudnn.operation_graph
// CHECK: }

// CHECK: @cudnn.tensor.create.4d.nhwc(i64, i64, i64, i64, i64) -> !cudnn.tensor
// CHECK: @cudnn.add(!cudnn.tensor, f32, !cudnn.tensor, f32, i32) -> !cudnn.tensor
// CHECK: @cudnn.operation_graph.create(!cudnn.tensor) -> !cudnn.operation_graph

// -----

cudnn.graph @bias(
  %x: !cudnn.tensor<8x32x4x4xf32, NHWC>,
  %b: !cudnn.tensor<1x32x1x1xf32, NHWC>
) -> !cudnn.tensor<8x32x4x4xf32, NHWC> {
  %0 = cudnn.bias(%x, %b)
    : (!cudnn.tensor<8x32x4x4xf32, NHWC>, !cudnn.tensor<1x32x1x1xf32, NHWC>)
    -> !cudnn.tensor<8x32x4x4xf32, NHWC>
  cudnn.return %0: !cudnn.tensor<8x32x4x4xf32, NHWC>
}

// CHECK: func.func @bias.builder() -> !cudnn.operation_graph {
// CHECK:   %[[X:.*]] = call @cudnn.tensor.create.4d.nhwc
// CHECK:   %[[B:.*]] = call @cudnn.tensor.create.4d.nhwc
// CHECK:   %[[VIRTUAL:.*]] = arith.constant 0 : i32
// CHECK:   %[[Y:.*]] = call @cudnn.bias(%[[X]], %[[B]], %[[VIRTUAL]])
// CHECK:   %[[GRAPH:.*]] = call @cudnn.operation_graph.create(%[[Y]])
// CHECK:   return %[[GRAPH]] : !cudnn.operation_graph
// CHECK: }

// CHECK: @cudnn.tensor.create.4d.nhwc(i64, i64, i64, i64, i64) -> !cudnn.tensor
// CHECK: @cudnn.bias(!cudnn.tensor, !cudnn.tensor, i32) -> !cudnn.tensor
// CHECK: @cudnn.operation_graph.create(!cudnn.tensor) -> !cudnn.operation_graph

// -----

cudnn.graph @convolution(
  %image: !cudnn.tensor<8x32x4x4xf32, NHWC>,
  %filter: !cudnn.tensor<32x32x1x1xf32, NHWC>
) -> !cudnn.tensor<8x32x4x4xf32, NHWC> {
  %0 = cudnn.convolution(%image, %filter) alpha=1.0 beta=0.0
         spatial_dim_count=2
         spatial_stride=[1,1]
         pre_padding=[1,1]
         post_padding=[1,1]
         dilation=[1,1]
    : (!cudnn.tensor<8x32x4x4xf32, NHWC>, !cudnn.tensor<32x32x1x1xf32, NHWC>)
    -> !cudnn.tensor<8x32x4x4xf32, NHWC>
  cudnn.return %0: !cudnn.tensor<8x32x4x4xf32, NHWC>
}

// CHECK: func.func @convolution.builder() -> !cudnn.operation_graph {
// CHECK:   %[[X:.*]] = call @cudnn.tensor.create.4d.nhwc
// CHECK:   %[[W:.*]] = call @cudnn.tensor.create.4d.nhwc
// CHECK:   %[[VIRTUAL:.*]] = arith.constant 0 : i32
// CHECK:   %[[Y:.*]] = call @cudnn.convolution.2d(%[[X]], %[[W]],
// CHECK:                                         %[[VIRTUAL]])
// CHECK:   %[[GRAPH:.*]] = call @cudnn.operation_graph.create(%[[Y]])
// CHECK:   return %[[GRAPH]] : !cudnn.operation_graph
// CHECK: }

// CHECK: @cudnn.tensor.create.4d.nhwc(i64, i64, i64, i64, i64) -> !cudnn.tensor
// CHECK: @cudnn.convolution.2d(!cudnn.tensor, !cudnn.tensor, i64, i64, i64,
// CHECK-SAME:                  i64, i64, i64, i64, i64, i32) -> !cudnn.tensor
// CHECK: @cudnn.operation_graph.create(!cudnn.tensor) -> !cudnn.operation_graph

// -----

cudnn.graph @graph(%arg: !cudnn.tensor<1x4x8xf32>) -> !cudnn.tensor<1x4x8xf32> {
  cudnn.return %arg: !cudnn.tensor<1x4x8xf32>
}

func.func @main(%arg: tensor<1x4x8xf32>) -> tensor<1x4x8xf32> {
  %0 = cudnn.call @graph(%arg) : (tensor<1x4x8xf32>) -> tensor<1x4x8xf32>
  return %0 : tensor<1x4x8xf32>
}

// CHECK: func.func @graph.builder() -> !cudnn.operation_graph

// CHECK: func.func @main(%[[ARG:.*]]: tensor<1x4x8xf32>) -> tensor<1x4x8xf32> {
// CHECK:   %[[G:.*]] = call @graph.builder()
// CHECK:   %[[E:.*]] = call @cudnn.executable.create(%[[G]])
// CHECK:   %[[BUF0:.*]] = hal.tensor.export %[[ARG]] "graph.arg.0"
// CHECK:   %[[BUF1:.*]] = call @cudnn.execute.1(%[[E]], %[[BUF0]])
// CHECK:   %[[RES:.*]] = hal.tensor.import %[[BUF1]] "graph.result"
// CHECK:   return %[[RES]] : tensor<1x4x8xf32>
// CHECK: }

// CHECK: @cudnn.executable.create(!cudnn.operation_graph) -> !cudnn.executable
// CHECK: @cudnn.execute.1(!cudnn.executable, !hal.buffer_view) -> !hal.buffer_view
