// RUN: iree-compile --iree-plugin=openxla-triton --iree-input-type=stablehlo  \
// RUN:              --iree-hal-target-backends=cuda %s                        \
// RUN: | iree-run-module --module=- --device=cuda --function=main             \
// RUN:                   --input=128xf32=2 --input=128xf32=6                  \
// RUN: | FileCheck %s

tt.func @add_inplace_kernel(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: i32) {
  %c64_i32 = arith.constant 64 : i32
  %0 = tt.get_program_id x : i32
  %1 = arith.muli %0, %c64_i32 : i32
  %2 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
  %3 = tt.splat %1 : (i32) -> tensor<64xi32>
  %4 = arith.addi %3, %2 : tensor<64xi32>
  %5 = tt.splat %arg2 : (i32) -> tensor<64xi32>
  %6 = arith.cmpi slt, %4, %5 : tensor<64xi32>
  %7 = tt.splat %arg0 : (!tt.ptr<f32>) -> tensor<64x!tt.ptr<f32>>
  %8 = tt.addptr %7, %4 : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
  %9 = tt.load %8, %6 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<64xf32>
  %10 = tt.splat %arg1 : (!tt.ptr<f32>) -> tensor<64x!tt.ptr<f32>>
  %11 = tt.addptr %10, %4 : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
  %12 = tt.load %11, %6 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<64xf32>
  %13 = arith.addf %9, %12 : tensor<64xf32>
  %14 = tt.splat %arg1 : (!tt.ptr<f32>) -> tensor<64x!tt.ptr<f32>>
  %15 = tt.addptr %14, %4 : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
  tt.store %15, %13, %6 : tensor<64xf32>
  tt.return
}

func.func @main(%arg0: tensor<?xf32>, %arg1: tensor<?xf32>) -> tensor<?xf32> {
  %c0 = arith.constant 0 : index
  %d0 = tensor.dim %arg0, %c0 : tensor<?xf32>
  %g0 = affine.apply affine_map<()[s0] -> (s0 ceildiv 64)>()[%d0]

  // Currently ABI only supports i32 scalars.
  %d0_i32 = arith.index_cast %d0 : index to i32

  %0 = triton.call @add_inplace_kernel[%g0](%arg0, %arg1, %d0_i32)
    : (tensor<?xf32>{%d0}, tensor<?xf32>{%d0}, i32) -> %arg1{%d0}

  return %0 : tensor<?xf32>
}

// CHECK: EXEC @main
// CHECK: result[0]: hal.buffer_view
// CHECK: 128xf32=8 8 8 8 8
