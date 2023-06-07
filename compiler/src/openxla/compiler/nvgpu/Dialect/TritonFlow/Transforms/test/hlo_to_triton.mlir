// RUN: BYTECODE=$(iree-opt $(dirname %s)/hlo_to_triton_bytecode.mlir          \
// RUN:              --iree-plugin=openxla-triton                              \
// RUN:              --emit-bytecode | od -v -t x1 -A n | tr -d ' \n')

// RUN: sed 's/#BYTECODE#/'"$BYTECODE"'/g' %s > %t-bytecode.mlir

// RUN: iree-opt %t-bytecode.mlir --iree-plugin=openxla-triton                 \
// RUN:             --split-input-file --openxla-nvgpu-convert-hlo-to-triton   \
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

// -----
// Check that operand alias gets lowered to a call with tied operand.

func.func @foo(%arg0: tensor<10xf32>, %arg1: tensor<10xf32>) -> tensor<10xf32> {
  %0 = mhlo.custom_call @__triton$call(%arg0, %arg1) {
    backend_config = "#BYTECODE#",
    grid = array<i64: 1, 1, 1>,
    num_warps = 4 : index,
    scalar_args_indices = array<i64: 0>,
    scalar_args_values = [10 : i32],
    output_operand_aliases = [#mhlo.output_operand_alias<
                                output_tuple_indices = [],
                                operand_index = 1,
                                operand_tuple_indices = []>]
  } : (tensor<10xf32>, tensor<10xf32>) -> tensor<10xf32>
  return %0 : tensor<10xf32>
}

// CHECK:  tt.func @triton(%{{.*}}: i32 {tt.foo}, %{{.*}}: !tt.ptr<f32>,
// CHECK:                  %{{.*}}: !tt.ptr<f32>)

// CHECK: func @foo(%[[ARG0:.*]]: tensor<10xf32>, %[[ARG1:.*]]: tensor<10xf32>)
// CHECK:   %[[RES:.*]] = triton.call @triton
// CHECK:                 ({{.*}}, %[[ARG0]], %[[ARG1]])
// CHECK:                 : (i32, tensor<10xf32>, tensor<10xf32>) -> %[[ARG1]]
// CHECK:   return %[[RES]] : tensor<10xf32>
// CHECK: }
