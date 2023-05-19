// RUN: iree-opt %s --iree-plugin=openxla-triton --split-input-file --verify-diagnostics

// expected-error @+1 {{op expected exactly one inner builtin.module operation}}
triton.executable @foo {
}

// -----

// expected-error @+1 {{op expected exactly one inner builtin.module operation}}
triton.executable @foo {
  builtin.module {}
  builtin.module {}
}

// -----

triton.executable @foo {
  // expected-error @+1 {{op refers to an unknown Triton function: @bar}}
  triton.executable.export @bar
  builtin.module {}
}

// -----

func.func @main(%arg0: index) {
  // expected-error @+1 {{op refers to an unknown Triton entry point: @foo}}
  triton.dispatch @foo[%arg0]() : () -> ()
  return
}

// -----

func.func @main(%arg0: index) {
  // expected-error @+1 {{op refers to an unknown Triton function: @foo}}
  triton.call @foo[%arg0]() : () -> ()
  return
}
