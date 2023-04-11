// Copyright 2023 The OpenXLA Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "openxla/runtime/nvgpu/cudnn_api.h"

#include <iree/base/assert.h>
#include <iree/base/internal/span.h>
#include <iree/base/status.h>
#include <iree/base/status_cc.h>
#include <iree/vm/ref_cc.h>
#include <openxla/runtime/nvgpu/status_util.h>

#include <type_traits>

#include "openxla/runtime/nvgpu/cudnn_stub.h"

namespace openxla::runtime::nvgpu {

using namespace iree;

using cudnn_frontend::TensorBuilder;

// clang-format off
#include "openxla/runtime/nvgpu/cudnn_stub.h.inc"
// clang-format on

//===----------------------------------------------------------------------===//
// CuDNNArgTensor
//===----------------------------------------------------------------------===//

CuDNNArgTensor::CuDNNArgTensor(openxla_cudnn_dynamic_symbols_t* syms,
                               cudnn_frontend::Tensor tensor)
    : CuDNNTensor(Kind::kArg), syms_(syms), tensor_(std::move(tensor)) {}

CuDNNArgTensor::~CuDNNArgTensor() {
  ScopedCuDNNStubs stubs(syms_);
  tensor_.reset();
}

const cudnn_frontend::Tensor& CuDNNArgTensor::tensor() const {
  return *tensor_;
}

//===----------------------------------------------------------------------===//
// CuDNNOpResultTensor
//===----------------------------------------------------------------------===//

CuDNNOpResultTensor::CuDNNOpResultTensor(openxla_cudnn_dynamic_symbols_t* syms,
                                         iree::span<CuDNNTensor* const> inputs,
                                         cudnn_frontend::Operation operation,
                                         cudnn_frontend::Tensor tensor)
    : CuDNNTensor(Kind::kOpResult),
      syms_(syms),
      operation_(std::move(operation)),
      tensor_(std::move(tensor)) {
  for (CuDNNTensor* input : inputs) {
    inputs_.push_back(vm::retain_ref(input));
  }
}

CuDNNOpResultTensor::~CuDNNOpResultTensor() {
  ScopedCuDNNStubs stubs(syms_);
  operation_.reset();
  tensor_.reset();
}

std::vector<CuDNNTensor*> CuDNNOpResultTensor::inputs() const {
  std::vector<CuDNNTensor*> ptrs;
  for (auto& input : inputs_) ptrs.push_back(input.get());
  return ptrs;
}

const cudnn_frontend::Operation* CuDNNOpResultTensor::operation() const {
  return &*operation_;
}

const cudnn_frontend::Tensor& CuDNNOpResultTensor::tensor() const {
  return *tensor_;
}

//===----------------------------------------------------------------------===//
// CuDNNOperationGraph
//===----------------------------------------------------------------------===//

CuDNNOperationGraph::CuDNNOperationGraph(openxla_cudnn_dynamic_symbols_t* syms,
                                         cudnn_frontend::OperationGraph graph)
    : syms_(syms), graph_(std::move(graph)) {}

CuDNNOperationGraph::~CuDNNOperationGraph() {
  ScopedCuDNNStubs stubs(syms_);
  graph_.reset();
}

cudnn_frontend::OperationGraph& CuDNNOperationGraph::graph() { return *graph_; }

//===----------------------------------------------------------------------===//
// CuDNNExecutable.
//===----------------------------------------------------------------------===//

CuDNNExecutable::CuDNNExecutable(
    openxla_cudnn_dynamic_symbols_t* syms, CuDNNOperationGraph& graph,
    span<const cudnn_frontend::ExecutionPlan> plans)
    : syms_(syms),
      graph_(vm::retain_ref(&graph)),
      plans_(plans.begin(), plans.end()) {}

CuDNNExecutable::~CuDNNExecutable() {
  ScopedCuDNNStubs stubs(syms_);
  plans_.clear();
}

//===----------------------------------------------------------------------===//
// Wrappers around cuDNN APIs export from a cuDNN module to the user
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// CreateArgument
//===----------------------------------------------------------------------===//

StatusOr<vm::ref<CuDNNTensor>> CreateArgument(
    openxla_cudnn_dynamic_symbols_t* syms, span<const int64_t> dims,
    span<const int64_t> strides, int64_t uid, cudnnDataType_t dtype,
    int64_t alignment) {
  ScopedCuDNNStubs stubs(syms);
  cudnn_frontend::Tensor tensor = cudnn_frontend::TensorBuilder()
                                      .setDim(dims.size(), dims.data())
                                      .setStride(strides.size(), strides.data())
                                      .setId(uid)
                                      .setAlignment(alignment)
                                      .setDataType(dtype)
                                      .build();
  IREE_RETURN_IF_ERROR(CUDNN_CONVERT_STATUS(syms, tensor.get_status()));
  return vm::ref<CuDNNTensor>(new CuDNNArgTensor(syms, std::move(tensor)));
}

//===----------------------------------------------------------------------===//
// CreatePointwiseRelu
//===----------------------------------------------------------------------===//

StatusOr<vm::ref<CuDNNTensor>> CreatePointwiseRelu(
    openxla_cudnn_dynamic_symbols_t* syms, CuDNNTensor& input,
    double lower_clip, double upper_clip, int64_t uid, int64_t alignment) {
  ScopedCuDNNStubs stubs(syms);

  // Prepare tensor descriptor for activation output.
  cudnn_frontend::Tensor tensor = cudnn_frontend::TensorBuilder()
                                      .cloneFrom(input.tensor(), uid)
                                      .setAlignment(alignment)
                                      .build();
  IREE_RETURN_IF_ERROR(CUDNN_CONVERT_STATUS(syms, tensor.get_status()));

  // Prepare activation descriptor.
  cudnn_frontend::PointWiseDesc activation =
      cudnn_frontend::PointWiseDescBuilder()
          .setMode(CUDNN_POINTWISE_RELU_FWD)
          .setClipping(lower_clip, upper_clip)
          .build();
  IREE_RETURN_IF_ERROR(CUDNN_CONVERT_STATUS(syms, activation.get_status()));

  // Create operation.
  cudnn_frontend::Operation operation =
      cudnn_frontend::OperationBuilder(
          CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
          .setxDesc(input.tensor())
          .setyDesc(tensor)
          .setpwDesc(activation)
          .build();
  IREE_RETURN_IF_ERROR(CUDNN_CONVERT_STATUS(syms, operation.get_status()));

  return vm::ref<CuDNNTensor>(new CuDNNOpResultTensor(
      syms, {&input}, std::move(operation), std::move(tensor)));
}

//===----------------------------------------------------------------------===//
// CreateConvolution
//===----------------------------------------------------------------------===//

static int64_t GetFwdConvDilatedFilterDim(int64_t filter_dim,
                                          int64_t dilation) {
  return ((filter_dim - 1) * dilation) + 1;
}

static int64_t GetFwdConvPaddedImageDim(int64_t tensor_dim, int64_t padding) {
  return tensor_dim + (2 * padding);
}

static int64_t GetFwdConvOutputDim(int64_t tensor_dim, int64_t padding,
                                   int64_t filter_dim, int64_t stride,
                                   int64_t dilation) {
  int64_t padded = GetFwdConvPaddedImageDim(tensor_dim, padding);
  int64_t dilated = GetFwdConvDilatedFilterDim(filter_dim, dilation);
  return ((padded - dilated) / stride) + 1;
}

StatusOr<vm::ref<CuDNNTensor>> CreateConvolution(
    openxla_cudnn_dynamic_symbols_t* syms, CuDNNTensor& input,
    CuDNNTensor& filter, int64_t uid, int64_t alignment) {
  ScopedCuDNNStubs stubs(syms);

  span<const int64_t> input_dims(input->getDim(), input->getDimCount());
  span<const int64_t> filter_dims(filter->getDim(), filter->getDimCount());

  // TODO(ezhulenev): Add support for 3-D convolutions.
  static constexpr int64_t kSpatialDims = 2;

  if (input_dims.size() != 4) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "3d convolution is not supported");
  }

  if (input_dims.size() != filter_dims.size()) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "convolution input and filter must have the same rank");
  }

  // TODO(ezhulenev): Add support for padded, dilated and strided convolutions.
  std::array<int64_t, kSpatialDims> paddings = {0, 0};
  std::array<int64_t, kSpatialDims> strides = {1, 1};
  std::array<int64_t, kSpatialDims> dilations = {1, 1};

  // Compute convolution output dimensions.
  std::vector<int64_t> output_dims = {input_dims[0], input_dims[1]};  // [N, C]
  for (int d = 0; d < kSpatialDims; ++d) {
    output_dims.push_back(GetFwdConvOutputDim(input_dims[d + 2], paddings[d],
                                              filter_dims[d + 2], strides[d],
                                              dilations[d]));
  }

  // Compute strides for output tensor based on input format.
  bool is_nhwc = input->getStride()[1] == 1;
  std::vector<int64_t> output_strides =
      is_nhwc ? GetChannelsLastStrides(output_dims)
              : GetRowMajorStrides(output_dims);

  // Prepare tensor descriptor for convolution output.
  cudnn_frontend::Tensor tensor =
      cudnn_frontend::TensorBuilder()
          .cloneFrom(input.tensor(), uid)
          .setAlignment(alignment)
          .setDim(output_dims.size(), output_dims.data())
          .setStride(output_strides.size(), output_strides.data())
          .build();
  IREE_RETURN_IF_ERROR(CUDNN_CONVERT_STATUS(syms, tensor.get_status()));

  // Prepare a forward convolution descriptor.
  cudnn_frontend::ConvDesc convolution =
      cudnn_frontend::ConvDescBuilder()
          .setComputeType(CUDNN_DATA_FLOAT)
          .setMathMode(CUDNN_CONVOLUTION)
          .setSpatialDimCount(kSpatialDims)
          .setSpatialStride(kSpatialDims, strides.data())
          .setPrePadding(kSpatialDims, paddings.data())
          .setPostPadding(kSpatialDims, paddings.data())
          .setDilation(kSpatialDims, dilations.data())
          .build();
  IREE_RETURN_IF_ERROR(CUDNN_CONVERT_STATUS(syms, convolution.get_status()));

  // Create operation.
  cudnn_frontend::Operation operation =
      cudnn_frontend::OperationBuilder(
          CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR)
          .setxDesc(input.tensor())
          .setwDesc(filter.tensor())
          .setyDesc(tensor)
          .setcDesc(convolution)
          .setAlpha(1.0)
          .setBeta(0.0)
          .build();
  IREE_RETURN_IF_ERROR(CUDNN_CONVERT_STATUS(syms, operation.get_status()));

  return vm::ref<CuDNNTensor>(new CuDNNOpResultTensor(
      syms, {&input}, std::move(operation), std::move(tensor)));
}

//===----------------------------------------------------------------------===//
// CreateOperationGraph
//===----------------------------------------------------------------------===//

template <typename To>
static To* DynCast(CuDNNTensor* tensor) {
  return To::classof(tensor) ? static_cast<To*>(tensor) : nullptr;
}

StatusOr<vm::ref<CuDNNOperationGraph>> CreateOperationGraph(
    openxla_cudnn_dynamic_symbols_t* syms, cudnnHandle_t handle,
    span<CuDNNTensor* const> results) {
  ScopedCuDNNStubs stubs(syms);

  // Collect cuDNN operations producing tensor results.
  std::vector<CuDNNTensor*> worklist(results.begin(), results.end());
  std::vector<const cudnn_frontend::Operation*> ops;

  // TODO(ezhulenev): Take care of duplicate operations when traversing a tensor
  // use-def chains (with an end-to-end test once we'll support them).

  while (!worklist.empty()) {
    CuDNNTensor* tensor = worklist.back();
    worklist.pop_back();

    // Add cudnn_frontend operation and follow inputs.
    if (auto* op_result = DynCast<CuDNNOpResultTensor>(tensor)) {
      ops.push_back(op_result->operation());
      std::vector<CuDNNTensor*> inputs = op_result->inputs();
      worklist.insert(worklist.end(), inputs.begin(), inputs.end());
    }
  }

  // Construct a cudnn_frontend operation graph.
  auto graph = cudnn_frontend::OperationGraphBuilder()
                   .setHandle(handle)
                   .setOperationGraph(ops.size(), ops.data())
                   .build();
  IREE_RETURN_IF_ERROR(CUDNN_CONVERT_STATUS(syms, graph.get_status()));

  return vm::ref<CuDNNOperationGraph>(
      new CuDNNOperationGraph(syms, std::move(graph)));
}

//===----------------------------------------------------------------------===//
// CreateExecutable.
//===----------------------------------------------------------------------===//

// TODO(ezhulenev): We need to be able to configure what engine configs should
// be supported by the cuDNN executable, e.g. skip all configs that non
// determenistic results (see CUDNN_NUMERICAL_NOTE_NONDETERMINISTIC note).
static bool AcceptAllGraphs(cudnnBackendDescriptor_t) { return false; }

iree::StatusOr<iree::vm::ref<CuDNNExecutable>> CreateExecutable(
    openxla_cudnn_dynamic_symbols_t* syms, cudnnHandle_t handle,
    CuDNNOperationGraph& graph) {
  ScopedCuDNNStubs stubs(syms);

  // Collect supported engine configs.
  cudnn_frontend::EngineConfigList configs;

  // TODO(ezhulenev): Heuristics should be configurable. Also it should be
  // configurable if fallback kernels should be enabled.
  std::vector<cudnnStatus_t> statuses = cudnn_frontend::get_heuristics_list<1>(
      {"heuristics_mode_a"}, graph.graph(),
      AcceptAllGraphs, configs);

  if (configs.empty()) {
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "cuDNN operation graph is not supported");
  }

  // Prepare execution plans for filtered engine configs. Not all configs
  // actually can be instantated as execution plans, some of them might be
  // unsupported at run time.
  std::vector<cudnn_frontend::ExecutionPlan> plans;
  for (auto config : configs) {
    cudnn_frontend::ExecutionPlan plan =
        cudnn_frontend::ExecutionPlanBuilder()
            .setHandle(handle)
            .setEngineConfig(config, graph.graph().getTag())
            .build();

    // Skip engine configs that are not supported by the current cuDNN version.ß
    if (plan.get_status() == CUDNN_STATUS_NOT_SUPPORTED) {
      continue;
    }

    IREE_RETURN_IF_ERROR(CUDNN_CONVERT_STATUS(syms, plan.get_status()));
    plans.push_back(std::move(plan));
  }

  // If we end up with empty execution plans, it means that current version of
  // cuDNN can't compiler the given operation graph.
  if (plans.empty()) {
    return iree_make_status(
        IREE_STATUS_INTERNAL,
        "didn't find any engine config supporting cuDNN operation graph");
  }

  return vm::ref<CuDNNExecutable>(new CuDNNExecutable(syms, graph, plans));
}

//===----------------------------------------------------------------------===//
// Helper functions for setting up cuDNN descriptors
//===----------------------------------------------------------------------===//

std::vector<int64_t> GetRowMajorStrides(span<const int64_t> dims) {
  std::vector<int64_t> strides(dims.size(), 1);
  for (int64_t d = dims.size() - 2; d >= 0; --d)
    strides[d] = dims[d] * strides[d + 1];
  return strides;
}

std::vector<int64_t> GetChannelsLastStrides(span<const int64_t> dims) {
  IREE_ASSERT(dims.size() == 4 || dims.size() == 5);
  std::vector<int64_t> strides(dims.size(), 1);
  strides[1] = 1;
  strides[dims.size() - 1] = strides[1] * dims[1];
  for (int64_t d = dims.size() - 2; d >= 2; --d) {
    strides[d] = strides[d + 1] * dims[d + 1];
  }
  strides[0] = strides[2] * dims[2];
  return strides;
}

}  // namespace openxla::runtime::nvgpu

//===----------------------------------------------------------------------===//
// Register types with IREE VM
//===----------------------------------------------------------------------===//

IREE_VM_DEFINE_TYPE_ADAPTERS(cudnn_tensor,
                             openxla::runtime::nvgpu::CuDNNTensor);
IREE_VM_DEFINE_TYPE_ADAPTERS(cudnn_operation_graph,
                             openxla::runtime::nvgpu::CuDNNOperationGraph);
IREE_VM_DEFINE_TYPE_ADAPTERS(cudnn_executable,
                             openxla::runtime::nvgpu::CuDNNExecutable);
