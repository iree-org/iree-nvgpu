// RUN: iree-opt %s --iree-plugin=openxla-triton --split-input-file            \
// RUN:             --openxla-nvgpu-outline-triton-calls                       \
// RUN:   | FileCheck %s

tt.func @triton(%arg0: i32, %arg1: !tt.ptr<f32>, %arg2: !tt.ptr<f32>) {
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

// CHECK: #[[LAYOUT:.*]] = #hal.pipeline.layout<push_constants = 1,
// CHECK:   sets = [<0, bindings = [
// CHECK:                 <0, storage_buffer, ReadOnly>,
// CHECK:                 <1, storage_buffer>
// CHECK:               ]>
// CHECK:          ]>

// CHECK: triton.executable public @triton.executable {
// CHECK:     triton.executable.export public @triton layout(#[[LAYOUT]])
// CHECK:     builtin.module {
// CHECK:       tt.func @triton({{.*}}) {
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
