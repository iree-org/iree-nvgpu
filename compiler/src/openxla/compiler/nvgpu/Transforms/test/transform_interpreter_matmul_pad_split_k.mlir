// RUN: iree-opt %s --iree-plugin=openxla-transform --openxla-nvgpu-flow-transform-interpreter='enable-matmul-pad-split-k=1' %s | FileCheck %s
// RUN: iree-opt %s --iree-plugin=openxla-transform --openxla-nvgpu-flow-transform-interpreter='enable-matmul-pad-split-k=1 debug-print-constructed-module=1' %s | FileCheck %s --check-prefix=SCRIPT

// SCRIPT: module @__transform attributes {transform.with_named_sequence}

// SCRIPT-LABEL: transform.named_sequence private @match_matmul_f32_133x133x128
// SCRIPT-SAME:    (%[[ARG0:.+]]: !transform.any_op {transform.readonly}) -> !transform.any_op
// SCRIPT:   %[[v0:.+]] = transform.match.structured failures(propagate) %[[ARG0]] : (!transform.any_op) -> !transform.any_op
// SCRIPT:   ^{{.*}}(%[[ARG1:.+]]: !transform.any_op):
// SCRIPT:      transform.match.operation_name %[[ARG1]] ["linalg.matmul"] : !transform.any_op
// SCRIPT:      %[[v1:.+]] = transform.match.structured.dim %[[ARG1]][0] : (!transform.any_op) -> !transform.param<i64>
// SCRIPT:      %[[v2:.+]] = transform.param.constant 133 : i64 -> !transform.param<i64>
// SCRIPT:      transform.match.param.cmpi eq %[[v1]], %[[v2]] : !transform.param<i64>
// SCRIPT:      %[[v3:.+]] = transform.match.structured.dim %[[ARG1]][1]
// SCRIPT:      %[[v4:.+]] = transform.param.constant 133
// SCRIPT:      transform.match.param.cmpi eq %[[v3]], %[[v4]]
// SCRIPT:      %[[v5:.+]] = transform.match.structured.dim %[[ARG1]][2]
// SCRIPT:      %[[v6:.+]] = transform.param.constant 128
// SCRIPT:      transform.match.param.cmpi eq %[[v5]], %[[v6]]
// SCRIPT:      transform.match.structured.yield %[[ARG1]]
// SCRIPT:    transform.yield %[[v0]] : !transform.any_op
  
// SCRIPT-LABEL: transform.named_sequence private @match_matmul_f32_514x130x500
// SCRIPT-LABEL: transform.named_sequence private @match_matmul_f32_515x131x512
 
// SCRIPT-LABEL: transform.named_sequence private @matmul_f32_pad
// SCRIPT-SAME:     (%[[ARG0:.+]]: !transform.any_op {transform.consumed})
// SCRIPT:         %[[v0]] = transform.structured.pad %[[ARG0]] {pack_paddings = [0, 0, 0], pad_to_multiple_of = [32, 32, 1728], padding_dimensions = [0, 1, 2], padding_values = [0.000000e+00 : f32, 0.000000e+00 : f32, 0.000000e+00 : f32]} : (!transform.any_op) -> !transform.any_op
// SCRIPT:         transform.yield 
  
// SCRIPT-LABEL: transform.named_sequence private @matmul_f32_split_k
// SCRIPT-SAME: (%[[ARG0:.+]]: !transform.any_op {transform.consumed})
// SCRIPT:        %[[v0:.+]] = transform.structured.pad %arg0 {pack_paddings = [0, 0, 0], pad_to_multiple_of = [32, 32, 1728], padding_dimensions = [0, 1, 2], padding_values = [0.000000e+00 : f32, 0.000000e+00 : f32, 0.000000e+00 : f32]} : (!transform.any_op) -> !transform.any_op
// SCRIPT:        %{{.*}}, %{{.*}}, %[[SPLIT:.+]], %{{.*}} = transform.structured.split_reduction %0 {insert_split_dimension = 2 : i64, split_factor = 108 : i64} : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
// SCRIPT:        transform.structured.interchange %[[SPLIT]] iterator_interchange = [2, 0, 1, 3] : (!transform.any_op) -> !transform.any_op
// SCRIPT:        transform.yield 
  
// SCRIPT:  transform.sequence  failures(propagate)
// SCRIPT:  ^{{.*}}(%[[ARG0:.+]]: !transform.any_op):
// SCRIPT:    transform.iree.register_match_callbacks
// SCRIPT:    transform.iree.register_nvgpu_match_callbacks
// SCRIPT:    foreach_match in %[[ARG0]]
// SCRIPT:        @match_matmul_f32_133x133x128 -> @matmul_f32_pad, 
// SCRIPT:        @match_matmul_f32_514x130x500 -> @matmul_f32_split_k, 
// SCRIPT:        @match_matmul_f32_515x131x512 -> @matmul_f32_split_k
  


!A_t = tensor<514x500xf32>
!B_t = tensor<500x130xf32>
!C_t = tensor<514x130xf32>

func.func @matmul_static(
    %A : !A_t, %B : !B_t, %C : !C_t) -> !C_t {
  %0 = linalg.matmul ins(%A, %B : !A_t, !B_t)
                     outs(%C : !C_t) -> !C_t
  return %0 : !C_t
}

// CHECK-LABEL: @matmul_static
// CHECK: %[[padded_lhs:.+]] = tensor.pad
// CHECK: %[[padded_rhs:.+]] = tensor.pad
// CHECK: %[[padded_res:.+]] = tensor.pad
// CHECK: %[[expanded_lhs:.+]] = tensor.expand_shape %[[padded_lhs]] {{.*}} : tensor<544x1728xf32> into tensor<544x108x16xf32>
// CHECK: %[[expanded_rhs:.+]] = tensor.expand_shape %[[padded_rhs]] {{.*}} : tensor<1728x160xf32> into tensor<108x16x160xf32>
// CHECK: linalg.generic
// CHECK-SAME: iterator_types = ["parallel", "parallel", "parallel", "reduction"]
// CHECK-SAME: ins(%[[expanded_lhs]], %[[expanded_rhs]]
// CHECK: linalg.generic
// CHECK-SAME: iterator_types = ["parallel", "parallel", "reduction"]
// CHECK-SAME: outs(%[[padded_res]]

func.func @fill_matmul_static(
    %A : !A_t, %B : !B_t, %C : !C_t) -> !C_t {
  %f0 = arith.constant 0.0 : f32
  %out = tensor.empty() : !C_t
  %filled = linalg.fill ins(%f0 : f32) outs(%out : !C_t) -> !C_t
  %0 = linalg.matmul ins(%A, %B : !A_t, !B_t)
                     outs(%filled : !C_t) -> !C_t
  return %0 : !C_t
}

// CHECK-LABEL: @fill_matmul_static
// CHECK: %[[padded_lhs:.+]] = tensor.pad
// CHECK: %[[padded_rhs:.+]] = tensor.pad
// CHECK: %[[padded_res:.+]] = tensor.pad
// CHECK: %[[expanded_lhs:.+]] = tensor.expand_shape %[[padded_lhs]] {{.*}} : tensor<544x1728xf32> into tensor<544x108x16xf32>
// CHECK: %[[expanded_rhs:.+]] = tensor.expand_shape %[[padded_rhs]] {{.*}} : tensor<1728x160xf32> into tensor<108x16x160xf32>
// CHECK: linalg.fill {{.*}} -> tensor<544x160x108xf32>
// CHECK: linalg.generic
// CHECK-SAME: iterator_types = ["parallel", "parallel", "parallel", "reduction"]
// CHECK-SAME: ins(%[[expanded_lhs]], %[[expanded_rhs]]
// CHECK: linalg.generic
// CHECK-SAME: iterator_types = ["parallel", "parallel", "reduction"]
// CHECK-SAME: outs(%[[padded_res]]

// -----

!A_t = tensor<133x128xf32>
!B_t = tensor<128x133xf32>
!C_t = tensor<133x133xf32>

// CHECK-LABEL: @matmul_static_pad
// CHECK-COUNT-3: tensor.pad
// CHECK: linalg.matmul
func.func @matmul_static_pad(
    %A : !A_t, %B : !B_t, %C : !C_t) -> !C_t {
  %0 = linalg.matmul ins(%A, %B : !A_t, !B_t)
                     outs(%C : !C_t) -> !C_t
  return %0 : !C_t
}

// -----


!A_t = tensor<128x128xf32>
!B_t = tensor<128x128xf32>
!C_t = tensor<128x128xf32>

// CHECK-LABEL: @matmul_static_nopad
// CHECK-NOT: tensor.pad
// CHECK: linalg.matmul
func.func @matmul_static_nopad(
    %A : !A_t, %B : !B_t, %C : !C_t) -> !C_t {
  %0 = linalg.matmul ins(%A, %B : !A_t, !B_t)
                     outs(%C : !C_t) -> !C_t
  return %0 : !C_t
}
