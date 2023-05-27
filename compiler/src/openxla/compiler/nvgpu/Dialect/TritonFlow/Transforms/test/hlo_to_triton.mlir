// RUN: export BYTECODE=$(iree-opt $(dirname %s)/emit_triton_bytecode.mlir \
// RUN:                     --iree-plugin=openxla-triton \
// RUN:                     --emit-bytecode | od -v -t x1 -A n | tr -d ' \n')

// RUN: sed 's/#BYTECODE#/'"$BYTECODE"'/g' %s > %t-bytecode.mlir

// RUN: iree-opt %t-bytecode.mlir --iree-plugin=openxla-triton                 \
// RUN:             --openxla-nvgpu-convert-hlo-to-triton                      \
// RUN:   | FileCheck %s --dump-input=always

func.func @foo(%arg0: tensor<10xf32>) -> tensor<10xf32> {
  %0 = mhlo.custom_call @__triton$call(%arg0) {
    backend_config = "#BYTECODE#",
    grid = array<i64: 1, 1, 1>,
    num_warps = 4 : index,
    scalar_args_indices = array<i64: 0>,
    scalar_args_values = [10 : i32]
  } : (tensor<10xf32>) -> tensor<10xf32>
  return %0 : tensor<10xf32>
}

// CHECK:  tt.func @triton(%{{.*}}: i32 {tt.foo}, %{{.*}}: !tt.ptr<f32>,
// CHECK:                  %{{.*}}: !tt.ptr<f32>)

// CHECK: func @foo(%[[ARG0:.*]]: tensor<10xf32>) -> tensor<10xf32> {
// CHECK:   %[[C10:.*]] = arith.constant 10 : i32
// CHECK:   %[[X:.*]] = arith.constant 1 : index
// CHECK:   %[[Y:.*]] = arith.constant 1 : index
// CHECK:   %[[Z:.*]] = arith.constant 1 : index
// CHECK:   %[[RES:.*]] = triton.call @triton[%[[X]], %[[Y]], %[[Z]]]
// CHECK:                 (%[[C10]], %[[ARG0]])
// CHECK:                 : (i32, tensor<10xf32>) -> tensor<10xf32>
// CHECK:   return %[[RES]] : tensor<10xf32>
// CHECK: }
