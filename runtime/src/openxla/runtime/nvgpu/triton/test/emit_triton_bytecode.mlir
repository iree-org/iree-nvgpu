// RUN: iree-opt %s --iree-plugin=openxla-triton --emit-bytecode \
// RUN:   | od -v -t x1 -A n | tr -d ' \n' | FileCheck %s

// CHECK: 4d4cef52034d4c495231372e302e30676974000131070105090103070311090b0d0f11
tt.func @add_kernel(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: i32, %arg3: !tt.ptr<f32>) {
  %c64_i32 = arith.constant 64 : i32
  %0 = tt.get_program_id {axis = 0 : i32} : i32
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
  %14 = tt.splat %arg3 : (!tt.ptr<f32>) -> tensor<64x!tt.ptr<f32>>
  %15 = tt.addptr %14, %4 : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
  tt.store %15, %13, %6 : tensor<64xf32>
  tt.return
}
