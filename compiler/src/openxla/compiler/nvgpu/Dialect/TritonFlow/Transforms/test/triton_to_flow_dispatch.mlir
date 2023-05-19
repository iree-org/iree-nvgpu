// RUN: iree-opt %s --iree-plugin=openxla-triton                               \
// RUN:             --openxla-nvgpu-convert-triton-to-flow-dispatch            \
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

// CHECK: #pipeline_layout = #hal.pipeline.layout<push_constants = 1,
// CHECK:   sets = [<0, bindings = [
// CHECK:                 <0, storage_buffer, ReadOnly>,
// CHECK:                 <1, storage_buffer, ReadOnly>,
// CHECK:                 <2, storage_buffer>
// CHECK:               ]>
// CHECK:          ]>

// CHECK: hal.executable.source private @triton.executable
// CHECK:   objects = #hal.executable.objects<{
// CHECK:     #executable_target_cuda_nvptx_fb = [
// CHECK:       #hal.executable.object<{path = "{{.*}}.ptx"}>
// CHECK:     ]
// CHECK:   }>

// CHECK: hal.executable.export public @triton ordinal(0)
// CHECK:   layout(#pipeline_layout)
// CHECK:   workgroup_size = [64 : index, 1 : index, 1 : index]

// CHECK: func @main(%[[ARG0:.*]]: tensor<?xf32>, %[[ARG1:.*]]: tensor<?xf32>)
// CHECK:   %[[DIM:.*]] = tensor.dim
// CHECK:   %[[GRID:.*]] = affine.apply
// CHECK:   flow.dispatch @triton.executable::@triton[%[[GRID]]]
// CHECK:     : (i32, tensor<?xf32>{%[[DIM]]}, tensor<?xf32>{%[[DIM]]})
// CHECK:     -> tensor<?xf32>{%[[DIM]]}
