// RUN: iree-opt %s --iree-plugin=openxla-triton                               \
// RUN:             --openxla-nvgpu-convert-triton-to-flow-dispatch            \
// RUN:   | FileCheck %s

#layout = #hal.pipeline.layout<push_constants = 1,
  sets = [<0, bindings = [<0, storage_buffer, ReadOnly>,
                          <1, storage_buffer>]>]>

triton.executable public @triton {
  triton.executable.export public @compute layout(#layout)
  builtin.module {
    tt.func @compute(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: i32) {
      tt.return
    }
  }
}

func.func @main(%idx: index, %i32: i32, %arg: tensor<?xf32>) -> tensor<?xf32> {
  %0 = triton.dispatch @triton::@compute[%idx](%arg, %i32)
       : (tensor<?xf32>{%idx}, i32) -> tensor<?xf32>{%idx}
  return %0 : tensor<?xf32>
}

// CHECK: #[[LAYOUT:.*]] = #hal.pipeline.layout
// CHECK:   push_constants = 1,
// CHECK:   <0, storage_buffer, ReadOnly>,
// CHECK:   <1, storage_buffer>
// CHECK-NOT: storage_buffer

// CHECK: hal.executable.source private @triton
// CHECK:   objects = #hal.executable.objects<{
// CHECK:     #executable_target_cuda_nvptx_fb = [
// CHECK:       #hal.executable.object<{path = "{{.*}}.ptx"}>
// CHECK:     ]
// CHECK:   }>

// CHECK: hal.executable.export public @compute ordinal(0)
// CHECK:   layout(#[[LAYOUT]])
// CHECK:   workgroup_size = [64 : index, 1 : index, 1 : index]

// CHECK: func @main(%[[ARG0:.*]]: index, %[[ARG1:.*]]: i32,
// CHECK:            %[[ARG2:.*]]: tensor<?xf32>)
// CHECK:   flow.dispatch @triton::@compute[%[[ARG0]]](%[[ARG2]], %[[ARG1]])
// CHECK:     : (tensor<?xf32>{%[[ARG0]]}, i32) -> tensor<?xf32>{%[[ARG0]]}
