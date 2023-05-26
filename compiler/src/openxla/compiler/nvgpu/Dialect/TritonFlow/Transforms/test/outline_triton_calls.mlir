// RUN: iree-opt %s --iree-plugin=openxla-triton --split-input-file            \
// RUN:             --openxla-nvgpu-outline-triton-calls                       \
// RUN:   | FileCheck %s

tt.func @triton(%arg0: i32 {tt.foo}, %arg1: !tt.ptr<f32>, %arg2: !tt.ptr<f32>) {
  tt.return
}

func.func @main(%idx: index, %i32: i32, %arg: tensor<?xf32>) -> tensor<?xf32> {
  %0 = triton.call @triton[%idx](%i32, %arg)
       : (i32, tensor<?xf32>{%idx}) -> tensor<?xf32>{%idx}

  return %0 : tensor<?xf32>
}

// CHECK: triton.executable public @triton.executable {
// CHECK:     triton.executable.export public @triton
// CHECK:     builtin.module {
// CHECK:       tt.func @triton(%{{.*}}: i32 {tt.foo}, %{{.*}}: !tt.ptr<f32>,
// CHECK:                       %{{.*}}: !tt.ptr<f32>)
// CHECK:         tt.return
// CHECK:       }
// CHECK:     }
// CHECK:   }

// CHECK: func @main(%[[ARG0:.*]]: index, %[[ARG1:.*]]: i32,
// CHECK:            %[[ARG2:.*]]: tensor<?xf32>) -> tensor<?xf32> {
// CHECK:   %[[RES:.*]] = triton.dispatch @triton.executable::@triton[%[[ARG0]]]
// CHECK:                 (%[[ARG1]], %[[ARG2]]) {{.*}} tensor<?xf32>{%[[ARG0]]}
// CHECK:   return %[[RES]] : tensor<?xf32>
// CHECK: }
