// RUN: iree-opt %s --iree-plugin=openxla-triton                               \
// RUN:             --openxla-nvgpu-convert-triton-to-custom-dispatch          \
// RUN:   | FileCheck %s

tt.func @add_kernel(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: !tt.ptr<f32>, %arg3: i32) {
  %c64_i32 = arith.constant 64 : i32
  %0 = tt.get_program_id {axis = 0 : i32} : i32
  %1 = arith.muli %0, %c64_i32 : i32
  %2 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
  %3 = tt.splat %1 : (i32) -> tensor<64xi32>
  %4 = arith.addi %3, %2 : tensor<64xi32>
  %5 = tt.splat %arg3 : (i32) -> tensor<64xi32>
  %6 = arith.cmpi slt, %4, %5 : tensor<64xi32>
  %7 = tt.splat %arg0 : (!tt.ptr<f32>) -> tensor<64x!tt.ptr<f32>>
  %8 = tt.addptr %7, %4 : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
  %9 = tt.load %8, %6 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<64xf32>
  %10 = tt.splat %arg1 : (!tt.ptr<f32>) -> tensor<64x!tt.ptr<f32>>
  %11 = tt.addptr %10, %4 : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
  %12 = tt.load %11, %6 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<64xf32>
  %13 = arith.addf %9, %12 : tensor<64xf32>
  %14 = tt.splat %arg2 : (!tt.ptr<f32>) -> tensor<64x!tt.ptr<f32>>
  %15 = tt.addptr %14, %4 : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
  tt.store %15, %13, %6 : tensor<64xf32>
  tt.return
}

func.func @main(%arg0: tensor<128xf32>, %arg1: tensor<128xf32>) -> tensor<128xf32> {
  %c0 = arith.constant 0 : index

  %dim = tensor.dim %arg0, %c0 : tensor<128xf32>
  %grid = affine.apply affine_map<()[s0] -> (s0 ceildiv 64)>()[%dim]
  %dim_i32 = arith.index_cast %dim : index to i32

  %0 = triton.dispatch @add_kernel[%grid](%dim_i32, %arg0, %arg1)
    : (i32, tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>

  return %0 : tensor<128xf32>
}

// CHECK: #pipeline_layout = #hal.pipeline.layout<push_constants = 1,
// CHECK:   sets = [<0, bindings = [
// CHECK:                 <0, storage_buffer, ReadOnly>,
// CHECK:                 <1, storage_buffer, ReadOnly>,
// CHECK:                 <2, storage_buffer>
// CHECK:               ]>
// CHECK:          ]>

// CHECK: hal.executable.source private @add_kernel.executable
// CHECK:   objects = #hal.executable.objects<{
// CHECK:     #executable_target_cuda_nvptx_fb = [
// CHECK:       #hal.executable.object<{path = "{{.*}}.ptx"}>
// CHECK:     ]
// CHECK:   }>

// CHECK: hal.executable.export public @add_kernel ordinal(0)
// CHECK:   layout(#pipeline_layout)
// CHECK:   workgroup_size = [64 : index, 1 : index, 1 : index]

// CHECK: func @main
// CHECK:   %[[GRID:.*]] = affine.apply
// CHECK:   flow.dispatch @add_kernel.executable::@add_kernel[%[[GRID]]]
