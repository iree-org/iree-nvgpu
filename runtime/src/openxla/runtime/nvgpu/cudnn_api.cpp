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
#include <iree/hal/buffer.h>
#include <iree/hal/drivers/cuda/cuda_buffer.h>
#include <iree/vm/ref_cc.h>
#include <openxla/runtime/nvgpu/status_util.h>

#include <iostream>
#include <type_traits>
#include <unordered_set>

#include "iree/hal/drivers/cuda/cuda_device.h"
#include "openxla/runtime/nvgpu/cudnn_stub.h"
#include "openxla/runtime/nvgpu/status_util.h"

namespace openxla::runtime::nvgpu {

using namespace iree;

using cudnn_frontend::TensorBuilder;

// clang-format off
#include "openxla/runtime/nvgpu/cudnn_stub.h.inc"
// clang-format on

static std::vector<CudnnTensor*> AsPtrs(span<const vm::ref<CudnnTensor>> refs) {
  std::vector<CudnnTensor*> ptrs;
  for (auto& ref : refs) ptrs.push_back(ref.get());
  return ptrs;
}

//===----------------------------------------------------------------------===//
// CudnnArgTensor
//===----------------------------------------------------------------------===//

CudnnArgTensor::CudnnArgTensor(openxla_cudnn_dynamic_symbols_t* syms,
                               cudnn_frontend::Tensor tensor)
    : CudnnTensor(Kind::kArg), syms_(syms), tensor_(std::move(tensor)) {}

CudnnArgTensor::~CudnnArgTensor() {
  ScopedCudnnStubs stubs(syms_);
  tensor_.reset();
}

const cudnn_frontend::Tensor& CudnnArgTensor::tensor() const {
  return *tensor_;
}

//===----------------------------------------------------------------------===//
// CudnnOpResultTensor
//===----------------------------------------------------------------------===//

CudnnOpResultTensor::CudnnOpResultTensor(openxla_cudnn_dynamic_symbols_t* syms,
                                         iree::span<CudnnTensor* const> inputs,
                                         cudnn_frontend::Operation operation,
                                         cudnn_frontend::Tensor tensor)
    : CudnnTensor(Kind::kOpResult),
      syms_(syms),
      operation_(std::move(operation)),
      tensor_(std::move(tensor)) {
  for (CudnnTensor* input : inputs) inputs_.push_back(vm::retain_ref(input));
}

CudnnOpResultTensor::~CudnnOpResultTensor() {
  ScopedCudnnStubs stubs(syms_);
  operation_.reset();
  tensor_.reset();
}

std::vector<CudnnTensor*> CudnnOpResultTensor::inputs() const {
  return AsPtrs(inputs_);
}

const cudnn_frontend::Operation* CudnnOpResultTensor::operation() const {
  return &*operation_;
}

const cudnn_frontend::Tensor& CudnnOpResultTensor::tensor() const {
  return *tensor_;
}

//===----------------------------------------------------------------------===//
// CudnnHandle
//===----------------------------------------------------------------------===//

CudnnHandle::CudnnHandle(openxla_cudnn_dynamic_symbols_t* syms,
                         vm::ref<iree_hal_device_t> device,
                         cudnnHandle_t handle)
    : syms_(syms), device_(std::move(device)), handle_(handle) {}

CudnnHandle::~CudnnHandle() {
  ScopedCudnnStubs stubs(syms_);
  CUDNN_STATUS_CHECK_OK(syms_, cudnnDestroy(handle_), "cudnnDestroy");
}

cudnnHandle_t CudnnHandle::handle() const { return handle_; }

iree_hal_device_t* CudnnHandle::device() const { return device_.get(); }

//===----------------------------------------------------------------------===//
// CudnnOperationGraph
//===----------------------------------------------------------------------===//

CudnnOperationGraph::CudnnOperationGraph(openxla_cudnn_dynamic_symbols_t* syms,
                                         CudnnHandle& handle,
                                         cudnn_frontend::OperationGraph graph,
                                         span<CudnnTensor* const> args,
                                         span<CudnnTensor* const> rets)
    : syms_(syms), handle_(vm::retain_ref(&handle)), graph_(std::move(graph)) {
  for (auto* arg : args) {
    args_.push_back(vm::retain_ref(arg));
    uids_.push_back(arg->tensor().getId());
  }
  for (auto* ret : rets) {
    rets_.push_back(vm::retain_ref(ret));
    uids_.push_back(ret->tensor().getId());
  }
}

CudnnOperationGraph::~CudnnOperationGraph() {
  ScopedCudnnStubs stubs(syms_);
  graph_.reset();
}

cudnn_frontend::OperationGraph& CudnnOperationGraph::graph() { return *graph_; }

std::vector<CudnnTensor*> CudnnOperationGraph::args() const {
  return AsPtrs(args_);
}

std::vector<CudnnTensor*> CudnnOperationGraph::rets() const {
  return AsPtrs(rets_);
}

iree::span<const int64_t> CudnnOperationGraph::uids() const { return uids_; }

cudnnHandle_t CudnnOperationGraph::handle() const { return handle_->handle(); }

iree_hal_device_t* CudnnOperationGraph::device() const {
  return handle_->device();
}

//===----------------------------------------------------------------------===//
// CudnnExecutable
//===----------------------------------------------------------------------===//

CudnnExecutable::CudnnExecutable(
    openxla_cudnn_dynamic_symbols_t* syms, CudnnOperationGraph& graph,
    span<const cudnn_frontend::ExecutionPlan> plans)
    : syms_(syms),
      graph_(vm::retain_ref(&graph)),
      plans_(plans.begin(), plans.end()) {}

CudnnExecutable::~CudnnExecutable() {
  ScopedCudnnStubs stubs(syms_);
  plans_.clear();
}

const CudnnOperationGraph& CudnnExecutable::graph() const { return *graph_; }

cudnnHandle_t CudnnExecutable::handle() const { return graph_->handle(); }

iree_hal_device_t* CudnnExecutable::device() const { return graph_->device(); }

// Converts IREE HAL buffers to CUDA device pointers.
static std::vector<CUdeviceptr> GetDevicePointers(
    span<iree_hal_buffer_t* const> buffers) {
  std::vector<CUdeviceptr> ptrs(buffers.size());
  for (size_t i = 0; i < buffers.size(); ++i) {
    iree_hal_buffer_t* allocated = iree_hal_buffer_allocated_buffer(buffers[i]);
    IREE_ASSERT_EQ(iree_hal_cuda_buffer_type(allocated),
                   IREE_HAL_CUDA_BUFFER_TYPE_DEVICE);

    ptrs[i] = iree_hal_cuda_buffer_device_pointer(allocated) +
              iree_hal_buffer_byte_offset(buffers[i]);
  }
  return ptrs;
}

Status CudnnExecutable::Execute(span<iree_hal_buffer_t* const> buffers) {
  ScopedCudnnStubs stubs(syms_);

  // Check that we have a buffer for every argument and result tensor.
  if (buffers.size() != (graph_->args().size() + graph_->rets().size())) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "number of buffer arguments doesn't match the "
                            "number of graph arguments and results");
  }

  auto ptrs = GetDevicePointers(buffers);
  static_assert(sizeof(CUdeviceptr) == sizeof(void*));

  auto uids = graph_->uids();
  IREE_ASSERT_EQ(ptrs.size(), uids.size());

  // TODO(ezhulenev): Support plans with workspace.
  IREE_ASSERT_EQ(plans_[0].getWorkspaceSize(), 0);
  void* workspace = nullptr;

  // Pack pointers to device buffers with unique tensor ids.
  cudnn_frontend::VariantPack pack =
      cudnn_frontend::VariantPackBuilder()
          .setWorkspacePointer(workspace)
          .setDataPointers(ptrs.size(), reinterpret_cast<void**>(ptrs.data()))
          .setUids(uids.size(), uids.data())
          .build();
  IREE_RETURN_IF_ERROR(CUDNN_CONVERT_STATUS(syms_, pack.get_status()));

  CUDNN_RETURN_IF_ERROR(syms_,
                        cudnnBackendExecute(handle(), plans_[0].get_raw_desc(),
                                            pack.get_raw_desc()),
                        "cudnnBackendExecute()");

  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Wrappers around cuDNN APIs export from a cuDNN module to the user
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// CreateTensor
//===----------------------------------------------------------------------===//

StatusOr<vm::ref<CudnnTensor>> CreateTensor(
    openxla_cudnn_dynamic_symbols_t* syms, span<const int64_t> dims,
    span<const int64_t> strides, int64_t uid, cudnnDataType_t dtype,
    int64_t alignment) {
  ScopedCudnnStubs stubs(syms);
  cudnn_frontend::Tensor tensor = cudnn_frontend::TensorBuilder()
                                      .setDim(dims.size(), dims.data())
                                      .setStride(strides.size(), strides.data())
                                      .setId(uid)
                                      .setAlignment(alignment)
                                      .setDataType(dtype)
                                      .build();
  IREE_RETURN_IF_ERROR(CUDNN_CONVERT_STATUS(syms, tensor.get_status()));
  return vm::ref<CudnnTensor>(new CudnnArgTensor(syms, std::move(tensor)));
}

//===----------------------------------------------------------------------===//
// CreatePointwiseRelu
//===----------------------------------------------------------------------===//

StatusOr<vm::ref<CudnnTensor>> CreatePointwiseRelu(
    openxla_cudnn_dynamic_symbols_t* syms, CudnnTensor& input,
    double lower_clip, double upper_clip, int64_t uid, int64_t alignment,
    bool is_virtual) {
  ScopedCudnnStubs stubs(syms);

  // Prepare tensor descriptor for activation output.
  cudnn_frontend::Tensor tensor = cudnn_frontend::TensorBuilder()
                                      .cloneFrom(input.tensor(), uid)
                                      .setAlignment(alignment)
                                      .setVirtual(is_virtual)
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

  return vm::ref<CudnnTensor>(new CudnnOpResultTensor(
      syms, {&input}, std::move(operation), std::move(tensor)));
}

//===----------------------------------------------------------------------===//
// CreatePointwiseUnary
//===----------------------------------------------------------------------===//

StatusOr<iree::vm::ref<CudnnTensor>> CreatePointwiseUnary(
    openxla_cudnn_dynamic_symbols_t* syms, cudnnPointwiseMode_t mode,
    CudnnTensor& x, float alpha, int64_t uid, int64_t alignment,
    bool is_virtual) {
  ScopedCudnnStubs stubs(syms);

  // Prepare tensor descriptor for the output.
  cudnn_frontend::Tensor tensor = cudnn_frontend::TensorBuilder()
                                      .cloneFrom(x.tensor(), uid)
                                      .setAlignment(alignment)
                                      .setVirtual(is_virtual)
                                      .build();
  IREE_RETURN_IF_ERROR(CUDNN_CONVERT_STATUS(syms, tensor.get_status()));

  // Prepare an operation descriptor.
  cudnn_frontend::PointWiseDesc desc = cudnn_frontend::PointWiseDescBuilder()
                                           .setMode(mode)
                                           .setComputeType(CUDNN_DATA_FLOAT)
                                           .build();
  IREE_RETURN_IF_ERROR(CUDNN_CONVERT_STATUS(syms, desc.get_status()));

  // Create a pointwise operation.
  cudnn_frontend::Operation operation =
      cudnn_frontend::OperationBuilder(
          CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
          .setxDesc(x.tensor())
          .setyDesc(tensor)
          .setpwDesc(desc)
          .setAlpha(alpha)
          .build();
  IREE_RETURN_IF_ERROR(CUDNN_CONVERT_STATUS(syms, operation.get_status()));

  return vm::ref<CudnnTensor>(new CudnnOpResultTensor(
      syms, {&x}, std::move(operation), std::move(tensor)));
}

//===----------------------------------------------------------------------===//
// CreatePointwiseBinary
//===----------------------------------------------------------------------===//

StatusOr<iree::vm::ref<CudnnTensor>> CreatePointwiseBinary(
    openxla_cudnn_dynamic_symbols_t* syms, cudnnPointwiseMode_t mode,
    CudnnTensor& x, float alpha, CudnnTensor& b, float alpha2, int64_t uid,
    int64_t alignment, bool is_virtual) {
  ScopedCudnnStubs stubs(syms);

  // TODO(ezhulenev): Pointwise operations in cuDNN do implicit broadcasting, so
  // in general it's unsafe to clone `x` for the output. We have to compute the
  // broadcasted shape with correct strides corresponding to the layout.

  // Prepare tensor descriptor for the output.
  cudnn_frontend::Tensor tensor = cudnn_frontend::TensorBuilder()
                                      .cloneFrom(x.tensor(), uid)
                                      .setAlignment(alignment)
                                      .setVirtual(is_virtual)
                                      .build();
  IREE_RETURN_IF_ERROR(CUDNN_CONVERT_STATUS(syms, tensor.get_status()));

  // Prepare an operation descriptor.
  cudnn_frontend::PointWiseDesc desc = cudnn_frontend::PointWiseDescBuilder()
                                           .setMode(mode)
                                           .setComputeType(CUDNN_DATA_FLOAT)
                                           .build();
  IREE_RETURN_IF_ERROR(CUDNN_CONVERT_STATUS(syms, desc.get_status()));

  // Create a pointwise operation.
  cudnn_frontend::Operation operation =
      cudnn_frontend::OperationBuilder(
          CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
          .setxDesc(x.tensor())
          .setbDesc(b.tensor())
          .setyDesc(tensor)
          .setpwDesc(desc)
          .setAlpha(alpha)
          .setAlpha2(alpha2)
          .build();
  IREE_RETURN_IF_ERROR(CUDNN_CONVERT_STATUS(syms, operation.get_status()));

  return vm::ref<CudnnTensor>(new CudnnOpResultTensor(
      syms, {&x, &b}, std::move(operation), std::move(tensor)));
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

StatusOr<vm::ref<CudnnTensor>> CreateConvolution(
    openxla_cudnn_dynamic_symbols_t* syms, CudnnTensor& input,
    CudnnTensor& filter, int64_t uid, int64_t alignment, bool is_virtual,
    cudnnConvolutionMode_t mode) {
  ScopedCudnnStubs stubs(syms);

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
          .setVirtual(is_virtual)
          .build();
  IREE_RETURN_IF_ERROR(CUDNN_CONVERT_STATUS(syms, tensor.get_status()));

  // Prepare a forward convolution descriptor.
  cudnn_frontend::ConvDesc convolution =
      cudnn_frontend::ConvDescBuilder()
          .setComputeType(CUDNN_DATA_FLOAT)
          .setMathMode(mode)
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

  return vm::ref<CudnnTensor>(new CudnnOpResultTensor(
      syms, {&input, &filter}, std::move(operation), std::move(tensor)));
}

//===----------------------------------------------------------------------===//
// CreateHandle
//===----------------------------------------------------------------------===//

StatusOr<vm::ref<CudnnHandle>> CreateHandle(
    openxla_cudnn_dynamic_symbols_t* syms, iree_hal_device_t* device) {
  IREE_ASSERT_ARGUMENT(device);
  cudnnHandle_t handle = {0};

  if (!iree_hal_cuda_device_isa(device))
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "HAL device must be a CUDA device to create cuDNN handle");

  CUcontext ctx = iree_hal_cuda_device_context(device);
  // TODO: We must guarantee that `ctx` is current when we create cuDNN
  // handle. Currently we rely on implicit guarantee that module is loaded
  // immediately after device is created, however it might not always be true?
  (void)ctx;  // Use something like `ScopedActivateContext` from XLA.

  CUDNN_RETURN_IF_ERROR(syms, cudnnCreate(&handle), "cudnnCreate");

  return vm::make_ref<CudnnHandle>(syms, vm::retain_ref(device), handle);
}

//===----------------------------------------------------------------------===//
// CreateOperationGraph
//===----------------------------------------------------------------------===//

template <typename To>
static To* DynCast(CudnnTensor* tensor) {
  return To::classof(tensor) ? static_cast<To*>(tensor) : nullptr;
}

StatusOr<vm::ref<CudnnOperationGraph>> CreateOperationGraph(
    openxla_cudnn_dynamic_symbols_t* syms, CudnnHandle& handle,
    span<CudnnTensor* const> rets) {
  ScopedCudnnStubs stubs(syms);

  // Tensors that should be passed as inputs when executing cuDNN graph.
  std::unordered_set<CudnnTensor*> args;

  // cuDNN operations defining the operation graph.
  std::vector<const cudnn_frontend::Operation*> ops;

  // TODO(ezhulenev): Take care of duplicate operations when traversing a tensor
  // use-def chains (with an end-to-end test once we'll support them).

  // Traverse cuDNN tensor use-def chains starting from returned tensors.
  std::vector<CudnnTensor*> worklist(rets.begin(), rets.end());
  while (!worklist.empty()) {
    CudnnTensor* tensor = worklist.back();
    worklist.pop_back();

    // Operation graph argument that must be passed as input.
    if (auto* arg = DynCast<CudnnArgTensor>(tensor)) {
      args.insert(arg);
    }

    // Add cudnn_frontend operation and follow inputs.
    if (auto* op_result = DynCast<CudnnOpResultTensor>(tensor)) {
      ops.push_back(op_result->operation());
      std::vector<CudnnTensor*> inputs = op_result->inputs();
      worklist.insert(worklist.end(), inputs.begin(), inputs.end());
    }
  }

  // Reverse collected operations to construct an operation graph tag starting
  // from the first compute operation in the graph.
  std::reverse(ops.begin(), ops.end());

  // Construct a cudnn_frontend operation graph.
  auto graph = cudnn_frontend::OperationGraphBuilder()
                   .setHandle(handle.handle())
                   .setOperationGraph(ops.size(), ops.data())
                   .build();
  IREE_RETURN_IF_ERROR(CUDNN_CONVERT_STATUS(syms, graph.get_status()));

  // Sort arguments by id, to get them in the same order as in `cudnn.graph`
  // operation signature.
  std::vector<CudnnTensor*> unique_args(args.begin(), args.end());
  std::sort(unique_args.begin(), unique_args.end(),
            [](CudnnTensor* a, CudnnTensor* b) {
              return a->tensor().getId() < b->tensor().getId();
            });

  return vm::ref<CudnnOperationGraph>(new CudnnOperationGraph(
      syms, handle, std::move(graph), unique_args, rets));
}

//===----------------------------------------------------------------------===//
// CreateExecutable
//===----------------------------------------------------------------------===//

// TODO(ezhulenev): We need to be able to configure what engine configs should
// be supported by the cuDNN executable, e.g. skip all configs that non
// determenistic results (see CUDNN_NUMERICAL_NOTE_NONDETERMINISTIC note).
static bool AcceptAllGraphs(cudnnBackendDescriptor_t) { return false; }

iree::StatusOr<iree::vm::ref<CudnnExecutable>> CreateExecutable(
    openxla_cudnn_dynamic_symbols_t* syms, CudnnOperationGraph& graph) {
  ScopedCudnnStubs stubs(syms);

  // Collect supported engine configs.
  cudnn_frontend::EngineConfigList configs;

  // TODO(ezhulenev): Heuristics should be configurable. Also it should be
  // configurable if fallback kernels should be enabled.
  std::vector<cudnnStatus_t> statuses = cudnn_frontend::get_heuristics_list<1>(
      {"heuristics_mode_a"}, graph.graph(), AcceptAllGraphs, configs);

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
            .setHandle(graph.handle())
            .setEngineConfig(config, graph.graph().getTag())
            .build();

    // Skip engine configs that are not supported by the current cuDNN version.
    if (plan.get_status() != CUDNN_STATUS_SUCCESS) {
      continue;
    }

    plans.push_back(std::move(plan));

    // TODO(ezhulenev): Currently we do not support any plan selection or auto
    // tuning, so we stop once we find the first supported plan.
    break;
  }

  // If we end up with empty execution plans, it means that current version of
  // cuDNN can't compiler the given operation graph.
  if (plans.empty()) {
    return iree_make_status(
        IREE_STATUS_INTERNAL,
        "didn't find any engine config supporting cuDNN operation graph");
  }

  return vm::ref<CudnnExecutable>(new CudnnExecutable(syms, graph, plans));
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
                             openxla::runtime::nvgpu::CudnnTensor);
IREE_VM_DEFINE_TYPE_ADAPTERS(cudnn_handle,
                             openxla::runtime::nvgpu::CudnnHandle);
IREE_VM_DEFINE_TYPE_ADAPTERS(cudnn_operation_graph,
                             openxla::runtime::nvgpu::CudnnOperationGraph);
IREE_VM_DEFINE_TYPE_ADAPTERS(cudnn_executable,
                             openxla::runtime::nvgpu::CudnnExecutable);
