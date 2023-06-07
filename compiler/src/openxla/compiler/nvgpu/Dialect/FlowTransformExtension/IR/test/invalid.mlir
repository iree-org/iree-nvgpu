// RUN: iree-opt %s --iree-plugin=openxla-transform --split-input-file --verify-diagnostics

transform.sequence failures(propagate) {
^bb0(%arg0: !transform.any_op):
  // expected-error @below {{expects the same number of operands and results}}
  transform.iree.filter_out_already_in_dispatch_region %arg0 : (!transform.any_op) -> ()
}
