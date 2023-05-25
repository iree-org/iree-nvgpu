// RUN: iree-opt --iree-plugin=openxla-triton --split-input-file %s \
// RUN:   | iree-opt --iree-plugin=openxla-triton --split-input-file \
// RUN:   | FileCheck %s

#layout = #hal.pipeline.layout<push_constants = 1,
  sets = [<0, bindings = [<0, storage_buffer>]>]>

triton.executable private @example {
  triton.executable.export @compute layout(#layout)
  builtin.module {
    func.func @compute(%arg0: !tt.ptr<f32>) { return }
  }
}

// CHECK: #[[LAYOUT:.*]] = #hal.pipeline.layout

// CHECK: triton.executable private @example {
// CHECK:   triton.executable.export public @compute layout(#[[LAYOUT]])
// CHECK:   builtin.module {
// CHECK:     func.func @compute(%[[ARG:.*]]: !tt.ptr<f32>)
// CHECK:   }
// CHECK: }

// -----

#layout = #hal.pipeline.layout<push_constants = 1,
  sets = [<0, bindings = [<0, storage_buffer>]>]>

triton.executable private @example {
  triton.executable.export public @compute as("foo") layout(#layout)
  builtin.module {
    func.func @compute(%arg0: !tt.ptr<f32>) { return }
  }
}

// CHECK: triton.executable.export public @compute as("foo")

// -----

#layout = #hal.pipeline.layout<push_constants = 1,
  sets = [<0, bindings = [<0, storage_buffer>]>]>

triton.executable private @example {
  triton.executable.export @compute layout(#layout)
  builtin.module {
    func.func @compute(%arg0: !tt.ptr<f32>) { return }
  }
}

func.func @main(%arg0: index, %arg1: tensor<4xf32>) {
  triton.dispatch @example::@compute[%arg0](%arg1) : (tensor<4xf32>) -> ()
  return
}

// CHECK: func @main(%[[ARG0:.*]]: index, %[[ARG1:.*]]: tensor<4xf32>)
// CHECK:   triton.dispatch @example::@compute[%arg0](%arg1)

// -----

#layout = #hal.pipeline.layout<push_constants = 1,
  sets = [<0, bindings = [<0, storage_buffer>]>]>

triton.executable private @example {
  triton.executable.export @compute layout(#layout)
  builtin.module {
    func.func @compute(%arg0: !tt.ptr<f32>) { return }
  }
}

func.func @main(%arg0: index, %arg1: tensor<?xf32>) {
  triton.dispatch @example::@compute[%arg0](%arg1)
    : (tensor<?xf32>{%arg0}) -> ()
  return
}

// CHECK: func @main(%[[ARG0:.*]]: index, %[[ARG1:.*]]: tensor<?xf32>)
// CHECK:   triton.dispatch @example::@compute[%[[ARG0]]](%[[ARG1]])
// CHECK      : (tensor<?xf32>{%[[ARG0]]) -> ()

// -----

#layout = #hal.pipeline.layout<push_constants = 1,
  sets = [<0, bindings = [<0, storage_buffer>]>]>

triton.executable private @example {
  triton.executable.export @compute layout(#layout)
  builtin.module {
    func.func @compute(%arg0: !tt.ptr<f32>) { return }
  }
}

func.func @main(%arg0: index, %arg1: tensor<?xf32>) {
  triton.dispatch @example::@compute[%arg0](%arg1)
    : (tensor<?xf32>{%arg0}) -> %arg1{%arg0}
  return
}

// CHECK: func @main(%[[ARG0:.*]]: index, %[[ARG1:.*]]: tensor<?xf32>)
// CHECK:   triton.dispatch @example::@compute[%[[ARG0]]](%[[ARG1]])
// CHECK      : (tensor<?xf32>{%[[ARG0]]) -> %[[ARG1]]{%[[ARG0]]}

// -----

tt.func private @triton(%arg0: !tt.ptr<f32>) {
  tt.return
}

func.func @main(%arg0: index, %arg1: tensor<4xf32>) {
  triton.call @triton[%arg0](%arg1) : (tensor<4xf32>) -> ()
  return
}

// CHECK: func @main(%[[ARG0:.*]]: index, %[[ARG1:.*]]: tensor<4xf32>)
// CHECK:   triton.call @triton[%[[ARG0]]](%[[ARG1]])

// -----

tt.func private @triton(%arg0: !tt.ptr<f32>) {
  tt.return
}

func.func @main(%arg0: index, %arg1: tensor<?xf32>) {
  triton.call @triton[%arg0](%arg1) : (tensor<?xf32>{%arg0}) -> ()
  return
}

// CHECK: func @main(%[[ARG0:.*]]: index, %[[ARG1:.*]]: tensor<?xf32>)
// CHECK:   triton.call @triton[%[[ARG0]]](%[[ARG1]])
// CHECK:     : (tensor<?xf32>{%[[ARG0]]}) -> ()

// -----

tt.func private @triton(%arg0: !tt.ptr<f32>) {
  tt.return
}

func.func @main(%arg0: index, %arg1: tensor<?xf32>) {
  triton.call @triton[%arg0](%arg1) : (tensor<?xf32>{%arg0}) -> %arg1{%arg0}
  return
}

// CHECK: func @main(%[[ARG0:.*]]: index, %[[ARG1:.*]]: tensor<?xf32>)
// CHECK:   triton.call @triton[%[[ARG0]]](%[[ARG1]])
// CHECK:     : (tensor<?xf32>{%[[ARG0]]}) -> %[[ARG1]]{%[[ARG0]]}

// -----

tt.func private @triton(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>) {
  tt.return
}

func.func @main(%arg0: index, %arg1: tensor<?xf32>) {
  triton.call @triton[%arg0](%arg1)
    : (tensor<?xf32>{%arg0}) -> (%arg1{%arg0}, tensor<?xf32>{%arg0})
  return
}

// CHECK: func @main(%[[ARG0:.*]]: index, %[[ARG1:.*]]: tensor<?xf32>)
// CHECK:   triton.call @triton[%[[ARG0]]](%[[ARG1]])
// CHECK:     : (tensor<?xf32>{%[[ARG0]]})
// CHECK:    -> (%[[ARG1]]{%[[ARG0]]}, tensor<?xf32>{%[[ARG0]]})

// -----

tt.func private @triton(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>) {
  tt.return
}

func.func @main(%arg0: index, %arg1: tensor<?xf32>) {
  triton.call @triton[%arg0](%arg1)
    : (tensor<?xf32>{%arg0}) -> (tensor<?xf32>{%arg0}, %arg1{%arg0})
  return
}

// CHECK: func @main(%[[ARG0:.*]]: index, %[[ARG1:.*]]: tensor<?xf32>)
// CHECK:   triton.call @triton[%[[ARG0]]](%[[ARG1]])
// CHECK:     : (tensor<?xf32>{%[[ARG0]]})
// CHECK:    -> (tensor<?xf32>{%[[ARG0]]}, %[[ARG1]]{%[[ARG0]]})
