// RUN: iree-opt %s --iree-plugin=openxla-transform --iree-transform-dialect-interpreter --split-input-file --verify-diagnostics

transform.sequence failures(propagate) {
^bb0(%arg0: !transform.any_op):
  %0 = transform.structured.match ops{["linalg.fill"]} in %arg0 : (!transform.any_op) -> !transform.any_op
  %1 = transform.iree.filter_out_already_in_dispatch_region %0 : (!transform.any_op) -> !transform.any_op
  transform.iree.emit_remark "after filtering" at %1 : !transform.any_op
}

func.func @foo() -> tensor<42xf32> {
  %c0 = arith.constant 0.0 : f32
  %0 = tensor.empty() : tensor<42xf32>
  // expected-remark @below {{after filtering}}
  %1 = linalg.fill ins(%c0 : f32) outs(%0 : tensor<42xf32>) -> tensor<42xf32>
  return %1 : tensor<42xf32>
}

// -----

transform.sequence failures(propagate) {
^bb0(%arg0: !transform.any_op):
  %0 = transform.structured.match ops{["linalg.fill"]} in %arg0 : (!transform.any_op) -> !transform.any_op
  %1 = transform.iree.filter_out_already_in_dispatch_region %0 : (!transform.any_op) -> !transform.any_op
  transform.iree.emit_remark "after filtering" at %1 : !transform.any_op
}

func.func @foo() -> tensor<42xf32> {
  %c0 = arith.constant 0.0 : f32
  %0 = tensor.empty() : tensor<42xf32>
  %1 = flow.dispatch.workgroups(%c0, %0) : (f32, tensor<42xf32>) -> %0 =
      (%arg1: f32, %arg2: !flow.dispatch.tensor<readwrite:tensor<42xf32>>) {
    %0 = flow.dispatch.tensor.load %arg2, offsets = [0], sizes = [42], strides = [1] : !flow.dispatch.tensor<readwrite:tensor<42xf32>> -> tensor<42xf32>
    // Not expecting a remark here.
    %1 = linalg.fill ins(%arg1 : f32) outs(%0 : tensor<42xf32>) -> tensor<42xf32>
    flow.dispatch.tensor.store %1, %arg2, offsets = [0], sizes = [42], strides = [1] : tensor<42xf32> -> !flow.dispatch.tensor<readwrite:tensor<42xf32>>
    flow.return
  }
  return %1 : tensor<42xf32>
}

// -----

transform.sequence failures(propagate) {
^bb0(%arg0: !transform.any_op):
  transform.iree.register_match_callbacks
  // expected-error @below {{callback 'reduction_partial' not found in the registry}}
  transform.iree.match_callback failures(suppress) "reduction_partial"(%arg0) 
    : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
}

// -----

transform.sequence failures(propagate) {
^bb0(%arg0: !transform.any_op):
  transform.iree.register_nvgpu_match_callbacks
  transform.iree.match_callback failures(suppress) "reduction_partial"(%arg0) 
    : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
}

// -----

transform.sequence failures(propagate) {
^bb0(%arg0: !transform.any_op):
  transform.iree.register_match_callbacks
  transform.iree.register_nvgpu_match_callbacks
  transform.iree.match_callback failures(suppress) "reduction_partial"(%arg0) 
    : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
}
