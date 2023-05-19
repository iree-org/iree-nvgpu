// RUN: iree-opt --iree-plugin=openxla-triton --split-input-file %s \
// RUN:   | iree-opt --iree-plugin=openxla-triton --split-input-file \
// RUN:   | FileCheck %s

triton.executable private @example {
  triton.executable.export @compute
  builtin.module {
    func.func @compute(%arg0: !tt.ptr<f32>) { return }
  }
}

// CHECK: triton.executable private @example {
// CHECK:   triton.executable.export public @compute
// CHECK:   builtin.module {
// CHECK:     func.func @compute(%[[ARG:.*]]: !tt.ptr<f32>)
// CHECK:   }
// CHECK: }

// -----

triton.executable private @example {
  triton.executable.export public @compute as("foo")
  builtin.module {
    func.func @compute(%arg0: !tt.ptr<f32>) { return }
  }
}

// CHECK: triton.executable.export public @compute as("foo")

// -----

triton.executable private @example {
  triton.executable.export @compute
  builtin.module {
    func.func @compute(%arg0: !tt.ptr<f32>) { return }
  }
}

func.func @main(%arg0: index, %arg1: tensor<?xf32>) {
  triton.dispatch @example::@compute[%arg0](%arg1) : (tensor<?xf32>) -> ()
  return
}

// CHECK: func @main(%[[ARG0:.*]]: index, %[[ARG1:.*]]: tensor<?xf32>)
// CHECK:   triton.dispatch @example::@compute[%arg0](%arg1)

// -----

func.func private @triton(%arg0: !tt.ptr<f32>) {
  return
}

func.func @main(%arg0: tensor<4xf32>) {
  %c1 = arith.constant 1 : index
  triton.call @triton[%c1](%arg0) : (tensor<4xf32>) -> ()
  return
}

// CHECK: func @main(%[[ARG:.*]]: tensor<4xf32>)
// CHECK:   %[[G:.*]] = arith.constant 1 : index
// CHECK:   triton.call @triton[%[[G]]](%[[ARG]])
