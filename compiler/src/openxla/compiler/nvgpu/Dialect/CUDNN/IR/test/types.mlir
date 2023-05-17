// RUN: iree-opt --iree-plugin=openxla-cudnn %s | \
// RUN: iree-opt --iree-plugin=openxla-cudnn | \
// RUN: FileCheck %s

#map = affine_map<(d0, d1, d2) -> (d0, d2, d1)>

// CHECK: @opaque(%arg0: !cudnn.tensor)
func.func @opaque(%arg0: !cudnn.tensor) {
  return
}

// CHECK: @rank0(%arg0: !cudnn.tensor<?x?x?xf32>)
func.func @rank0(%arg0: !cudnn.tensor<?x?x?xf32>) {
  return
}

// CHECK: @nhwc(%arg0: !cudnn.tensor<?x?x?x?xf32, NHWC>)
func.func @nhwc(%arg0: !cudnn.tensor<?x?x?x?xf32, NHWC>) {
  return
}

// CHECK: @nchw(%arg0: !cudnn.tensor<?x?x?x?xf32, NCHW>)
func.func @nchw(%arg0: !cudnn.tensor<?x?x?x?xf32, NCHW>) {
  return
}

// CHECK: @strided(
// CHECK:   %arg0: !cudnn.tensor<?x?x?xf32, affine_map<(d0, d1, d2) -> (d0, d2, d1)>>
// CHECK:   %arg1: !cudnn.tensor<?x?x?xf32, affine_map<(d0, d1, d2) -> (d0, d2, d1)>>
// CHECK: )
func.func @strided(
    %arg0: !cudnn.tensor<?x?x?xf32, affine_map<(d0, d1, d2) -> (d0, d2, d1)>>,
    %arg1: !cudnn.tensor<?x?x?xf32, #map>
) {
  return
}
