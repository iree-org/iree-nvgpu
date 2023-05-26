// RUN: iree-opt %s --iree-plugin=openxla-triton --split-input-file            \
// RUN:             --openxla-nvgpu-refine-triton-abi                          \
// RUN:   | FileCheck %s

triton.executable private @triton {
  triton.executable.export @foo
  builtin.module {
    tt.func @foo(%idx: !tt.ptr<f32>) { tt.return }
  }
}

func.func @main(%idx: index, %arg1: tensor<f32>) {
  triton.dispatch @triton::@foo[%idx](%arg1) : (tensor<f32>) -> ()
  return
}

// CHECK: #[[LAYOUT:.*]] = #hal.pipeline.layout
// CHECK:   push_constants = 0
// CHECK:   <0, storage_buffer, ReadOnly>
// CHECK-NOT: storage_buffer

// CHECK: triton.executable
// CHECK:   triton.executable.export public @foo layout(#[[LAYOUT]])
// CHECK:   tt.func @foo({{.*}}: !tt.ptr<f32>)

// CHECK: func @main(%[[ARG0:.*]]: index, %[[ARG1:.*]]: tensor<f32>)
// CHECK:   triton.dispatch @triton::@foo[%[[ARG0]]](%[[ARG1]])

// -----

triton.executable private @triton {
  triton.executable.export @foo
  builtin.module {
    tt.func @foo(%idx: !tt.ptr<f32>) { tt.return }
  }
}

func.func @main(%idx: index) {
  triton.dispatch @triton::@foo[%idx]() : () -> (tensor<f32>)
  return
}

// CHECK: #[[LAYOUT:.*]] = #hal.pipeline.layout
// CHECK:   push_constants = 0
// CHECK:   <0, storage_buffer>
// CHECK-NOT: storage_buffer

// CHECK: triton.executable
// CHECK:   triton.executable.export public @foo layout(#[[LAYOUT]])
// CHECK:   tt.func @foo({{.*}}: !tt.ptr<f32>)

// CHECK: func @main(%[[ARG0:.*]]: index)
// CHECK:   triton.dispatch @triton::@foo[%[[ARG0]]]()

// -----

triton.executable private @triton {
  triton.executable.export @foo
  builtin.module {
    tt.func @foo(%arg0: i32 {tt.foo}, %arg1: !tt.ptr<f32>) { tt.return }
  }
}

func.func @main(%idx: index, %arg1: i32) {
  triton.dispatch @triton::@foo[%idx](%arg1) : (i32) -> (tensor<f32>)
  return
}

// CHECK: #[[LAYOUT:.*]] = #hal.pipeline.layout
// CHECK:   push_constants = 1
// CHECK:   <0, storage_buffer>
// CHECK-NOT: storage_buffer

// CHECK: triton.executable
// CHECK:   triton.executable.export public @foo layout(#[[LAYOUT]])
// CHECK:   tt.func @foo({{.*}}: !tt.ptr<f32>, {{.*}}: i32 {tt.foo})

// CHECK: func @main(%[[ARG0:.*]]: index, %[[ARG1:.*]]: i32)
// CHECK:   triton.dispatch @triton::@foo[%[[ARG0]]](%[[ARG1]])

// -----

triton.executable private @triton {
  triton.executable.export @foo
  builtin.module {
    tt.func @foo(%arg0: i32, %arg1: !tt.ptr<f32>) { tt.return }
  }
}

func.func @main(%idx: index, %i32: i32, %arg: tensor<f32>) {
  triton.dispatch @triton::@foo[%idx](%i32, %arg) : (i32, tensor<f32>) -> ()
  return
}

// CHECK: #[[LAYOUT:.*]] = #hal.pipeline.layout
// CHECK:   push_constants = 1
// CHECK:   <0, storage_buffer, ReadOnly>
// CHECK-NOT: storage_buffer

// CHECK: triton.executable
// CHECK:   triton.executable.export public @foo layout(#[[LAYOUT]])
// CHECK:   tt.func @foo({{.*}}: !tt.ptr<f32>, {{.*}}: i32)

// CHECK: func @main(%[[ARG0:.*]]: index, %[[ARG1:.*]]: i32,
// CHECK:            %[[ARG2:.*]]: tensor<f32>)
// CHECK:   triton.dispatch @triton::@foo[%[[ARG0]]](%[[ARG2]], %[[ARG1]])

// -----

triton.executable private @triton {
  triton.executable.export @foo
  builtin.module {
    tt.func @foo(%arg0: i32, %arg1: !tt.ptr<f32>) { tt.return }
  }
}

func.func @main(%idx: index, %i32: i32, %t: tensor<f32>) {
  triton.dispatch @triton::@foo[%idx](%i32, %t) : (i32, tensor<f32>) -> %t
  return
}

// CHECK: #[[LAYOUT:.*]] = #hal.pipeline.layout
// CHECK:   push_constants = 1
// CHECK:   <0, storage_buffer>
// CHECK-NOT: storage_buffer

// CHECK: triton.executable
// CHECK:   triton.executable.export public @foo layout(#[[LAYOUT]])
// CHECK:   tt.func @foo({{.*}}: !tt.ptr<f32>, {{.*}}: i32)

// CHECK: func @main(%[[ARG0:.*]]: index, %[[ARG1:.*]]: i32,
// CHECK:            %[[ARG2:.*]]: tensor<f32>)
// CHECK:   triton.dispatch @triton::@foo[%[[ARG0]]](%[[ARG2]], %[[ARG1]])
// CHECK:   : (tensor<f32>, i32) -> %[[ARG2]]

// -----

triton.executable private @triton {
  triton.executable.export @foo
  builtin.module {
    tt.func @foo(%arg0: i32, %arg1: !tt.ptr<f32>, %arg2: !tt.ptr<i32>) {
      tt.return
    }
  }
}

func.func @main(%idx: index, %i32: i32, %t: tensor<f32>) {
  triton.dispatch @triton::@foo[%idx](%i32, %t)
    : (i32, tensor<f32>) -> (%t, tensor<i32>)
  return
}

// CHECK: #[[LAYOUT:.*]] = #hal.pipeline.layout
// CHECK:   push_constants = 1
// CHECK:   <0, storage_buffer>,
// CHECK:   <1, storage_buffer>
// CHECK-NOT: storage_buffer

// CHECK: triton.executable
// CHECK:   triton.executable.export public @foo layout(#[[LAYOUT]])
// CHECK:   tt.func @foo({{.*}}: !tt.ptr<f32>, {{.*}}: !tt.ptr<i32>,
// CHECK:                {{.*}}: i32)

// CHECK: func @main(%[[ARG0:.*]]: index, %[[ARG1:.*]]: i32,
// CHECK:            %[[ARG2:.*]]: tensor<f32>)
// CHECK:   triton.dispatch @triton::@foo[%[[ARG0]]](%[[ARG2]], %[[ARG1]])
// CHECK:   : (tensor<f32>, i32) -> (%[[ARG2]], tensor<i32>)
