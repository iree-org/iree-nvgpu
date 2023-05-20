// RUN: iree-opt %s --iree-plugin=openxla-cublas --split-input-file            \
// RUN:   | iree-opt --iree-plugin=openxla-cublas --split-input-file           \
// RUN:   | FileCheck %s

func.func @main(%arg0: !hal.device) -> !cublas.handle {
  %0 = cublas.handle(%arg0) : !cublas.handle
  return %0 : !cublas.handle
}

// CHECK: func @main(%[[ARG0:.*]]: !hal.device) -> !cublas.handle {
// CHECK:   cublas.handle(%[[ARG0]]) : !cublas.handle
// CHECK: }
