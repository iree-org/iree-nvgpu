// RUN: iree-opt %s --iree-plugin=openxla-triton                               \
// RUN:             --openxla-nvgpu-outline-triton-calls                       \
// RUN:   | FileCheck %s

tt.func @triton(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>,
                %arg2: !tt.ptr<f32>, %arg3: i32) {
  tt.return
}

func.func @main(%arg0: tensor<?xf32>, %arg1: tensor<?xf32>) -> tensor<?xf32> {
  %c0 = arith.constant 0 : index
  %d0 = tensor.dim %arg0, %c0 : tensor<?xf32>
  %g0 = affine.apply affine_map<()[s0] -> (s0 ceildiv 64)>()[%d0]

  // Currently ABI only supports i32 scalars.
  %d0_i32 = arith.index_cast %d0 : index to i32

  %0 = triton.call @triton[%g0](%d0_i32, %arg0, %arg1)
    : (i32, tensor<?xf32>{%d0}, tensor<?xf32>{%d0}) -> tensor<?xf32>{%d0}

  return %0 : tensor<?xf32>
}

// CHECK: triton.executable public @triton.executable {
// CHECK:     triton.executable.export public @triton
// CHECK:     builtin.module {
// CHECK:       tt.func @triton({{.*}}) {
// CHECK:         tt.return
// CHECK:       }
// CHECK:     }
// CHECK:   }

// CHECK: func @main({{.*}}) -> tensor<?xf32> {
// CHECK:   %[[RES:.*]] = triton.dispatch @triton.executable::@triton
// CHECK:   return %[[RES]] : tensor<?xf32>
// CHECK: }
