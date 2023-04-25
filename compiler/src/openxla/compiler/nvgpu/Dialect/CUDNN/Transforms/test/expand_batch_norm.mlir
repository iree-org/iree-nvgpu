// RUN: iree-opt %s --iree-plugin=openxla_nvgpu --split-input-file \
// RUN:   --pass-pipeline='builtin.module(openxla-nvgpu-expand-cudnn-operations)' \
// RUN:   | FileCheck %s

cudnn.graph @batch_norm_inference(
  %x: !cudnn.tensor<8x32x4x4xf32, NHWC>,
  %scale: !cudnn.tensor<1x32x1x1xf32, NHWC> {cudnn.scale},
  %offset: !cudnn.tensor<1x32x1x1xf32, NHWC> {cudnn.offset},
  %mean: !cudnn.tensor<1x32x1x1xf32, NHWC> {cudnn.mean},
  %var: !cudnn.tensor<1x32x1x1xf32, NHWC> {cudnn.var},
  %epsilon: !cudnn.tensor<1x1x1x1xf32, NHWC>
) -> !cudnn.tensor<8x32x4x4xf32, NHWC> {
  %0 = cudnn.batch_norm_inference(%x, %scale, %offset, %mean, %var, %epsilon)
    : (!cudnn.tensor<8x32x4x4xf32, NHWC>, !cudnn.tensor<1x32x1x1xf32, NHWC>,
       !cudnn.tensor<1x32x1x1xf32, NHWC>, !cudnn.tensor<1x32x1x1xf32, NHWC>,
       !cudnn.tensor<1x32x1x1xf32, NHWC>, !cudnn.tensor<1x1x1x1xf32, NHWC>)
    -> !cudnn.tensor<8x32x4x4xf32, NHWC>
  cudnn.return %0: !cudnn.tensor<8x32x4x4xf32, NHWC>
}

// CHECK: cudnn.graph @batch_norm_inference(
// CHECK:   %[[X:.*]]: !cudnn.tensor<8x32x4x4xf32, NHWC>,
// CHECK:   %[[SCALE:.*]]: !cudnn.tensor<1x32x1x1xf32, NHWC> {cudnn.scale}
// CHECK:   %[[OFFSET:.*]]: !cudnn.tensor<1x32x1x1xf32, NHWC> {cudnn.offset}
// CHECK:   %[[MEAN:.*]]: !cudnn.tensor<1x32x1x1xf32, NHWC> {cudnn.mean}
// CHECK:   %[[VAR:.*]]: !cudnn.tensor<1x32x1x1xf32, NHWC> {cudnn.var}
// CHECK:   %[[EPSILON:.*]]: !cudnn.tensor<1x1x1x1xf32, NHWC>
// CHECK: ) -> !cudnn.tensor<8x32x4x4xf32, NHWC> {
// CHECK:   %[[CENTERED:.*]] = cudnn.sub(%[[X]], %[[MEAN]])
// CHECK:   %[[ADD:.*]] = cudnn.add(%[[VAR]], %[[EPSILON]])
// CHECK:   %[[STDDEV:.*]] = cudnn.sqrt(%[[ADD]])
// CHECK:   %[[NORMALIZED:.*]] = cudnn.div(%[[CENTERED]], %[[STDDEV]])
// CHECK:   %[[SCALED:.*]] = cudnn.mul(%[[NORMALIZED]], %[[SCALE]])
// CHECK:   %[[SHIFTED:.*]] = cudnn.add(%[[SCALED]], %[[OFFSET]])
// CHECK:   cudnn.return %[[SHIFTED]]
// CHECK: }
