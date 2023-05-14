// RUN: iree-opt --iree-plugin=openxla-triton --split-input-file %s \
// RUN:   | iree-opt --iree-plugin=openxla-triton --split-input-file \
// RUN:   | FileCheck %s

func.func private @triton_func(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>,
                               %arg2: i32, %arg3: !tt.ptr<f32>) {
  return
}

func.func @main(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32>{
  %c1_idx = arith.constant 1 : index
  %c1_i32 = arith.constant 1 : i32
  %0 = triton.dispatch @triton_func[%c1_idx](%arg0, %arg1, %c1_i32)
       : (tensor<4xf32>, tensor<4xf32>, i32) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// CHECK: func @main(%[[ARG0:.*]]: tensor<4xf32>, %[[ARG1:.*]]: tensor<4xf32>)
// CHECK:   %[[G:.*]] = arith.constant 1 : index
// CHECK:   %[[D:.*]] = arith.constant 1 : i32
// CHECK:   triton.dispatch @triton_func[%[[G]]](%[[ARG0]], %[[ARG1]], %[[D]])
