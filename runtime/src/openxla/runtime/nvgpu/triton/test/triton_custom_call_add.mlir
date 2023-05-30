// RUN: BYTECODE=$(iree-opt $(dirname %s)/triton_add_bytecode.mlir             \
// RUN:                     --iree-plugin=openxla-triton                       \
// RUN:                     --emit-bytecode | od -v -t x1 -A n | tr -d ' \n')

// RUN: sed 's/#BYTECODE#/'"$BYTECODE"'/g' %s > %t-bytecode.mlir

// RUN: iree-compile --iree-plugin=openxla-triton --iree-input-type=mhlo       \
// RUN:              --iree-hal-target-backends=cuda %t-bytecode.mlir          \
// RUN: | iree-run-module --module=- --device=cuda --function=main             \
// RUN:                   --input=128xf32=2 --input=128xf32=6                  \
// RUN: | FileCheck %s

func.func @main(%arg0: tensor<128xf32>, %arg1: tensor<128xf32>) -> tensor<128xf32> {
  %0 = mhlo.custom_call @__triton$call(%arg0, %arg1) {
    backend_config = "#BYTECODE#",
    grid = array<i64: 1, 1, 1>,
    num_warps = 4 : index,
    scalar_args_indices = array<i64: 2>,
    scalar_args_values = [128 : i32]
  } : (tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
  return %0 : tensor<128xf32>
}

// CHECK: EXEC @main
// CHECK: result[0]: hal.buffer_view
// CHECK: 128xf32=8 8 8 8 8
