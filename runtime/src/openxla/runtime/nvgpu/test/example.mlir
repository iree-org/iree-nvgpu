// RUN: iree-compile %s --iree-hal-target-backends=cuda | openxla-runner - example.main | FileCheck %s

module @example {

  //===--------------------------------------------------------------------===//
  // Import functions from the cuDNN module.
  //===--------------------------------------------------------------------===//

  func.func private @cudnn.tensor.arg(
    %dtype: i64, %dims: !util.list<i64>, %uid: i64, %alignment: i64
  ) -> !cudnn.tensor

  func.func private @cudnn.tensor.arg.nhwc(
    %dtype: i64, %dims: !util.list<i64>, %uid: i64, %alignment: i64
  ) -> !cudnn.tensor

  func.func private @cudnn.pointwise_relu(
    %input: !cudnn.tensor, %lower: f32, %upper: f32, %uid: i64, %alignment: i64
  ) -> !cudnn.tensor

  func.func private @cudnn.convolution(
    %input: !cudnn.tensor, %filter: !cudnn.tensor, %uid: i64, %alignment: i64
  ) -> !cudnn.tensor

  func.func private @cudnn.graph.create(
    %tensor: !cudnn.tensor
  ) -> !cudnn.operation_graph
  
  func.func private @cudnn.executable.create(
    %tensor: !cudnn.operation_graph
  ) -> !cudnn.executable

  func.func private @cudnn.debug.tensor(
    %tensor: !cudnn.tensor
  )

  func.func private @cudnn.debug.graph(
    %graph: !cudnn.operation_graph
  )

  //===--------------------------------------------------------------------===//
  // Build and execute cuDNN graph.
  //===--------------------------------------------------------------------===//

  func.func @main() {
    %rank = arith.constant 4 : index

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index

    // CUDNN_DATA_FLOAT
    %dtype = arith.constant 0 : i64

    // Tensor alignment
    %align = arith.constant 32 : i64

    // Dimension values
    %d32 = arith.constant 32 : i64
    %d8 = arith.constant 8 : i64
    %d4 = arith.constant 4 : i64
    %d1 = arith.constant 1 : i64

    // Tensor UIDs
    %uid0 = arith.constant 0 : i64 // input
    %uid1 = arith.constant 1 : i64 // filter
    %uid2 = arith.constant 1 : i64 // output

    // Input: [8, 32, 4, 4]
    %input_dims = util.list.create %rank : !util.list<i64>
    util.list.resize %input_dims, %rank : !util.list<i64>
    util.list.set %input_dims[%c0], %d8 : !util.list<i64>
    util.list.set %input_dims[%c1], %d32 : !util.list<i64>
    util.list.set %input_dims[%c2], %d4 : !util.list<i64>
    util.list.set %input_dims[%c3], %d4 : !util.list<i64>

    // Filter: [32, 32, 2, 2]
    %filter_dims = util.list.create %rank : !util.list<i64>
    util.list.resize %filter_dims, %rank : !util.list<i64>
    util.list.set %filter_dims[%c0], %d32 : !util.list<i64>
    util.list.set %filter_dims[%c1], %d32 : !util.list<i64>
    util.list.set %filter_dims[%c2], %d1 : !util.list<i64>
    util.list.set %filter_dims[%c3], %d1 : !util.list<i64>

    // Tensor arguments
    %input = call @cudnn.tensor.arg.nhwc(%dtype, %input_dims, %uid0, %align)
               : (i64, !util.list<i64>, i64, i64) -> !cudnn.tensor
    %filter = call @cudnn.tensor.arg.nhwc(%dtype, %filter_dims, %uid1, %align)
               : (i64, !util.list<i64>, i64, i64) -> !cudnn.tensor

    // Create convolution operation
    %conv = call @cudnn.convolution(%input, %filter, %uid2, %align)
              : (!cudnn.tensor, !cudnn.tensor, i64, i64) -> !cudnn.tensor

    // Debug all cuDNN tensors in the graph
    call @cudnn.debug.tensor(%input) : (!cudnn.tensor) -> ()
    call @cudnn.debug.tensor(%filter) : (!cudnn.tensor) -> ()
    call @cudnn.debug.tensor(%conv) : (!cudnn.tensor) -> ()

    // Create an operation graph computing convolution/
    %graph = call @cudnn.graph.create(%conv)
           : (!cudnn.tensor) -> !cudnn.operation_graph

    // CHECK: Graph: CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR
    call @cudnn.debug.graph(%graph) : (!cudnn.operation_graph) -> ()

    // Create executable from an operation graph
    %executable = call @cudnn.executable.create(%graph)
           : (!cudnn.operation_graph) -> !cudnn.executable

    return
  }

}
