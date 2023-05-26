// RUN: iree-opt %s --iree-plugin=openxla-triton --split-input-file            \
// RUN:             --openxla-nvgpu-outline-triton-calls                       \
// RUN:   | FileCheck %s

tt.func @triton(%arg0: i32 {tt.foo}, %arg1: !tt.ptr<f32>, %arg2: !tt.ptr<f32>) {
  tt.return
}

func.func @main(%arg0: tensor<?xf32>) -> tensor<?xf32> {
  %c0 = arith.constant 0 : index
  %d0 = tensor.dim %arg0, %c0 : tensor<?xf32>
  %g0 = affine.apply affine_map<()[s0] -> (s0 ceildiv 64)>()[%d0]

  // Currently ABI only supports i32 scalars.
  %d0_i32 = arith.index_cast %d0 : index to i32

  %0 = triton.call @triton[%g0](%d0_i32, %arg0)
    : (i32, tensor<?xf32>{%d0}) -> tensor<?xf32>{%d0}

  return %0 : tensor<?xf32>
}

// CHECK: #[[LAYOUT:.*]] = #hal.pipeline.layout
// CHECK:   push_constants = 1
// CHECK:   <0, storage_buffer, ReadOnly>
// CHECK:   <1, storage_buffer>
// CHECK-NOT: storage_buffer

// CHECK: triton.executable public @triton.executable {
// CHECK:     triton.executable.export public @triton layout(#[[LAYOUT]])
// CHECK:     builtin.module {
// CHECK:       tt.func @triton(%{{.*}}: !tt.ptr<f32>, %{{.*}}: !tt.ptr<f32>,
// CHECK:                       %{{.*}}: i32 {tt.foo})
// CHECK:         tt.return
// CHECK:       }
// CHECK:     }
// CHECK:   }

// CHECK: func @main(%[[ARG0:.*]]: tensor<?xf32>) -> tensor<?xf32> {
// CHECK:   %[[G0:.*]] = affine.apply
// CHECK:   %[[I32:.*]] = arith.index_cast
// CHECK:   %[[RES:.*]] = triton.dispatch @triton.executable::@triton[%[[G0]]]
// CHECK:                                 (%[[ARG0]], %[[I32]])
// CHECK:   return %[[RES]] : tensor<?xf32>
// CHECK: }

// -----
// Check outlining Triton functions with tied results.

tt.func @triton(%arg0: i32 {tt.foo}, %arg1: !tt.ptr<f32>) {
  tt.return
}

func.func @main(%arg0: index, %arg1: tensor<?xf32>) -> tensor<?xf32> {
  %i32 = arith.index_cast %arg0 : index to i32
  %0 = triton.call @triton[%arg0](%i32, %arg1)
       : (i32, tensor<?xf32>{%arg0}) -> %arg1{%arg0}
  return %0 : tensor<?xf32>
}

// CHECK: #hal.pipeline.layout
// CHECK:   push_constants = 1
// CHECK:   <0, storage_buffer>
// CHECK-NOT: storage_buffer

// CHECK: triton.executable
// CHECK:   builtin.module
// CHECK:     tt.func @triton(%{{.*}}: !tt.ptr<f32>, %{{.*}}: i32 {tt.foo})

// CHECK: func @main(%[[ARG0:.*]]: index, %[[ARG1:.*]]: tensor<?xf32>)
// CHECK:   %[[I32:.*]] = arith.index_cast
// CHECK:   %[[RES:.*]] = triton.dispatch @triton.executable::@triton[%[[ARG0]]]
// CHECK:                   (%[[ARG1]], %[[I32]]) {{.*}} -> %[[ARG1]]{%[[ARG0]]}
// CHECK:   return %[[RES]] : tensor<?xf32>
// CHECK: }

// -----
// Check outlining Triton functions with tied and regular results.

tt.func @triton(%arg0: i32, %arg1: !tt.ptr<f32>, %arg2: !tt.ptr<i32>) {
  tt.return
}

func.func @main(%arg0: index, %arg1: tensor<?xf32>) {
  %i32 = arith.index_cast %arg0 : index to i32
  %0:2 = triton.call @triton[%arg0](%i32, %arg1)
         : (i32, tensor<?xf32>{%arg0}) -> (%arg1{%arg0}, tensor<?xi32>{%arg0})
  return
}

// CHECK: #hal.pipeline.layout
// CHECK:   push_constants = 1
// CHECK:   <0, storage_buffer>,
// CHECK:   <1, storage_buffer>
// CHECK-NOT: storage_buffer

// CHECK: triton.executable
// CHECK:   builtin.module
// CHECK:     tt.func @triton(%{{.*}}: !tt.ptr<f32>, %{{.*}}: !tt.ptr<i32>,
// CHECK:                     %{{.*}}: i32)

// CHECK: func @main(%[[ARG0:.*]]: index, %[[ARG1:.*]]: tensor<?xf32>)
// CHECK:   %[[I32:.*]] = arith.index_cast
// CHECK:   triton.dispatch @triton.executable::@triton[%[[ARG0]]]
// CHECK:      (%[[ARG1]], %[[I32]])
// CHECK:   -> (%[[ARG1]]{%[[ARG0]]}, tensor<?xi32>{%[[ARG0]]}
