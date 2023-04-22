// Copyright 2023 The OpenXLA Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "openxla/runtime/nvgpu/cudnn_module.h"

#include <iree/base/assert.h>
#include <iree/base/config.h>
#include <iree/base/internal/dynamic_library.h>
#include <iree/base/status.h>
#include <iree/base/status_cc.h>
#include <iree/hal/buffer.h>
#include <iree/hal/drivers/cuda/api.h>
#include <iree/hal/drivers/cuda/dynamic_symbols.h>
#include <iree/hal/drivers/cuda/status_util.h>
#include <iree/vm/list.h>
#include <iree/vm/ref_cc.h>
#include <openxla/runtime/nvgpu/cudnn_api.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>

#include "iree/hal/drivers/cuda/cuda_device.h"
#include "iree/modules/hal/types.h"
#include "iree/vm/native_module_cc.h"
#include "openxla/runtime/nvgpu/dynamic_symbols.h"
#include "openxla/runtime/nvgpu/status_util.h"

namespace openxla::runtime::nvgpu {

using namespace iree;

//===----------------------------------------------------------------------===//
// Helper functions for tensor layouts
//===----------------------------------------------------------------------===//

template <size_t n>
static std::array<int64_t, n> ToArr(span<const int64_t> span) {
  std::array<int64_t, n> arr;
  std::copy_n(span.begin(), n, arr.begin());
  return arr;
}

struct RowMajor {
  template <size_t n>
  static std::array<int64_t, n> strides(std::array<int64_t, n> dims) {
    return ToArr<n>(GetRowMajorStrides(dims));
  }
};

struct NHWC {
  static std::array<int64_t, 4> strides(std::array<int64_t, 4> dims) {
    return ToArr<4>(GetChannelsLastStrides(dims));
  }
};

// Converts cuDNN tensor shape + strides into a row-major shape that can be
// used to construct HAL buffer views.
static std::vector<iree_host_size_t> GetRowMajorShape(
    const cudnn_frontend::Tensor& tensor) {
  const int64_t* dim = tensor.getDim();
  const int64_t* stride = tensor.getStride();

  std::vector<int64_t> indices(tensor.getDimCount());
  std::iota(indices.begin(), indices.end(), 0);
  std::sort(indices.begin(), indices.end(),
            [&](int64_t a, int64_t b) { return stride[a] > stride[b]; });

  std::vector<iree_host_size_t> dims(tensor.getDimCount());
  for (size_t i = 0; i < indices.size(); ++i) dims[i] = dim[indices[i]];

  return dims;
}

//===----------------------------------------------------------------------===//
// CuDNN module state encapsulates all the state required for running cuDNN
// operations (launching cuDNN graphs on a stream) at run time.
//===----------------------------------------------------------------------===//

class CuDNNModuleState {
  // TODO(ezhulenev): This is a random value and never really verified at run
  // time. We always have to check that buffers we see at run time are properly
  // aligned according to what we promised to cuDNN.
  static constexpr int64_t kAlignment = 32;

 public:
  CuDNNModuleState(iree_hal_device_t* device, iree_allocator_t host_allocator,
                   iree_hal_cuda_dynamic_symbols_t cuda_syms,
                   openxla_cudnn_dynamic_symbols_t syms, cudnnHandle_t handle);
  ~CuDNNModuleState();

  enum class TensorFormat { kRowMajor, kChannelsLast };

  // Creates a new tensor with given shape and layout.
  template <size_t rank, typename Layout = RowMajor>
  StatusOr<vm::ref<CuDNNTensor>> TensorCreate(
      int64_t dtype, std::array<int64_t, rank> dimensions);

  // Creates a cuDNN operation graph computing `tensor` result.
  StatusOr<vm::ref<CuDNNOperationGraph>> OperationGraphCreate(
      const vm::ref<CuDNNTensor> tensor);

  // Creates a cuDNN executable from the given operation graph.
  StatusOr<vm::ref<CuDNNExecutable>> Executable(
      const vm::ref<CuDNNOperationGraph> graph);

  // Executes cuDNN executable with given HAL buffer view inputs and returns
  // result as a HAL buffer view.
  template <size_t n>
  StatusOr<vm::ref<iree_hal_buffer_view_t>> Execute(
      const vm::ref<CuDNNExecutable> executable,
      std::array<vm::ref<iree_hal_buffer_view_t>, n> inputs);

  // Creates a pointwise relu operation and returns result tensor.
  StatusOr<vm::ref<CuDNNTensor>> PointwiseRelu(const vm::ref<CuDNNTensor> input,
                                               float lower_clip,
                                               float upper_clip, int64_t uid,
                                               int64_t alignment,
                                               int32_t is_virtual);

  // Creates an add operation and returns a result tensor.
  StatusOr<vm::ref<CuDNNTensor>> Add(const vm::ref<CuDNNTensor> x, float alpha,
                                     const vm::ref<CuDNNTensor> b, float alpha2,
                                     int32_t is_virtual);

  // Creates a bias operation and returns a result tensor.
  StatusOr<vm::ref<CuDNNTensor>> Bias(const vm::ref<CuDNNTensor> x,
                                      const vm::ref<CuDNNTensor> b,
                                      int32_t is_virtual);

  // Creates a convolution operation and returns result tensor.
  template <size_t spatial_dims>
  StatusOr<vm::ref<CuDNNTensor>> Convolution(
      const vm::ref<CuDNNTensor> x, const vm::ref<CuDNNTensor> w,
      std::array<int64_t, spatial_dims> stride,
      std::array<int64_t, spatial_dims> pre_padding,
      std::array<int64_t, spatial_dims> post_padding,
      std::array<int64_t, spatial_dims> dilation, int32_t is_virtual);

  // Prints tensor debug information to stderr.
  Status PrintTensorDebug(const vm::ref<CuDNNTensor> tensor);

  // Prints graph debug information to stderr.
  Status PrintGraphDebug(const vm::ref<CuDNNOperationGraph> graph);

 private:
  CuDNNModuleState(const CuDNNModuleState&) = delete;
  CuDNNModuleState& operator=(const CuDNNModuleState&) = delete;

  iree_hal_device_t* device_;
  iree_allocator_t host_allocator_;

  iree_hal_cuda_dynamic_symbols_t cuda_syms_;
  openxla_cudnn_dynamic_symbols_t syms_;

  // IREE custom module state must be thread-compatible, and access to the same
  // state object will be synchronized by the caller, so we can safely access
  // cuDNN handle without any additional synchronization.
  cudnnHandle_t handle_;

  // We use automatic uid assignment for all cuDNN tensors in the graph.
  uint64_t uid = 0;
};

CuDNNModuleState::CuDNNModuleState(iree_hal_device_t* device,
                                   iree_allocator_t host_allocator,
                                   iree_hal_cuda_dynamic_symbols_t cuda_syms,
                                   openxla_cudnn_dynamic_symbols_t syms,
                                   cudnnHandle_t handle)
    : device_(device),
      host_allocator_(host_allocator),
      cuda_syms_(cuda_syms),
      syms_(syms),
      handle_(handle) {}

CuDNNModuleState::~CuDNNModuleState() {
  CUDNN_STATUS_CHECK_OK(&syms_, cudnnDestroy(handle_));
}

static StatusOr<cudnnDataType_t> ToCudnnDataType(int64_t dtype) {
  if (dtype < CUDNN_DATA_FLOAT || dtype > CUDNN_DATA_FAST_FLOAT_FOR_FP8)
    return Status(StatusCode::kInvalidArgument, "unsupported data type");
  return static_cast<cudnnDataType_t>(dtype);
}

template <size_t rank, typename Layout>
StatusOr<vm::ref<CuDNNTensor>> CuDNNModuleState::TensorCreate(
    int64_t dtype, std::array<int64_t, rank> dimensions) {
  IREE_ASSIGN_OR_RETURN(cudnnDataType_t data_type, ToCudnnDataType(dtype));
  std::array<int64_t, rank> strides = Layout::strides(dimensions);
  return CreateTensor(&syms_, dimensions, strides, uid++, data_type,
                      kAlignment);
}

Status CuDNNModuleState::PrintTensorDebug(const vm::ref<CuDNNTensor> tensor) {
  std::string desc = tensor->tensor().describe();
  fprintf(stderr, "Tensor: %s\n", desc.c_str());
  return OkStatus();
}

Status CuDNNModuleState::PrintGraphDebug(
    const vm::ref<CuDNNOperationGraph> graph) {
  std::string desc = graph->graph().describe();
  fprintf(stderr, "Graph: %s\n", desc.c_str());
  return OkStatus();
}

StatusOr<vm::ref<CuDNNTensor>> CuDNNModuleState::PointwiseRelu(
    const vm::ref<CuDNNTensor> input, float lower_clip, float upper_clip,
    int64_t uid, int64_t alignment, int32_t is_virtual) {
  return CreatePointwiseRelu(&syms_, *input, lower_clip, upper_clip, uid,
                             alignment, is_virtual);
}

StatusOr<vm::ref<CuDNNTensor>> CuDNNModuleState::Add(
    const vm::ref<CuDNNTensor> x, float alpha, const vm::ref<CuDNNTensor> b,
    float alpha2, int32_t is_virtual) {
  return CreateAdd(&syms_, *x, alpha, *b, alpha2, uid++, kAlignment,
                   is_virtual);
}

StatusOr<vm::ref<CuDNNTensor>> CuDNNModuleState::Bias(
    const vm::ref<CuDNNTensor> x, const vm::ref<CuDNNTensor> b,
    int32_t is_virtual) {
  return CreateBias(&syms_, *x, *b, uid++, kAlignment, is_virtual);
}

template <size_t spatial_dims>
StatusOr<vm::ref<CuDNNTensor>> CuDNNModuleState::Convolution(
    const vm::ref<CuDNNTensor> x, const vm::ref<CuDNNTensor> w,
    std::array<int64_t, spatial_dims> stride,
    std::array<int64_t, spatial_dims> pre_padding,
    std::array<int64_t, spatial_dims> post_padding,
    std::array<int64_t, spatial_dims> dilation, int32_t is_virtual) {
  return CreateConvolution(&syms_, *x, *w, uid++, kAlignment, is_virtual);
}

StatusOr<vm::ref<CuDNNOperationGraph>> CuDNNModuleState::OperationGraphCreate(
    const vm::ref<CuDNNTensor> tensor) {
  return CreateOperationGraph(&syms_, handle_, {tensor.get()});
}

StatusOr<vm::ref<CuDNNExecutable>> CuDNNModuleState::Executable(
    const vm::ref<CuDNNOperationGraph> graph) {
  return CreateExecutable(&syms_, handle_, *graph);
}

template <size_t n>
StatusOr<vm::ref<iree_hal_buffer_view_t>> CuDNNModuleState::Execute(
    const vm::ref<CuDNNExecutable> executable,
    std::array<vm::ref<iree_hal_buffer_view_t>, n> inputs) {
  // Arguments and results defined by the operation graph.
  std::vector<CuDNNTensor*> args = executable->graph().args();
  std::vector<CuDNNTensor*> rets = executable->graph().rets();
  IREE_ASSERT_EQ(rets.size(), 1);

  // Tensors required for running single convolution operation.
  const cudnn_frontend::Tensor& output = rets[0]->tensor();

  // Allocate buffer for tensor output.
  iree_hal_buffer_params_t output_buffer_params = {
      /*.usage=*/IREE_HAL_BUFFER_USAGE_DEFAULT | IREE_HAL_BUFFER_USAGE_MAPPING,
      /*.access=*/IREE_HAL_MEMORY_ACCESS_ALL,
      /*.type=*/IREE_HAL_MEMORY_TYPE_OPTIMAL_FOR_DEVICE |
          IREE_HAL_MEMORY_TYPE_HOST_VISIBLE,
      /*.queue_affinity=*/IREE_HAL_QUEUE_AFFINITY_ANY,
      /*.min_alignment=*/static_cast<iree_host_size_t>(output.getAlignment()),
  };

  vm::ref<iree_hal_semaphore_t> semaphore;
  IREE_RETURN_IF_ERROR(iree_hal_semaphore_create(device_, 0, &semaphore));
  vm::ref<iree_hal_fence_t> alloca_fence;
  IREE_RETURN_IF_ERROR(iree_hal_fence_create_at(
      semaphore.get(), 1, host_allocator_, &alloca_fence));

  // TODO(ezhulenev): Add support for all cuDNN data types.
  IREE_ASSERT_EQ(output.getDataType(), CUDNN_DATA_FLOAT);
  int64_t output_byte_length = output.getPackedElementCount() * sizeof(float);

  vm::ref<iree_hal_buffer_t> output_buffer;
  IREE_RETURN_IF_ERROR(iree_hal_device_queue_alloca(
      device_, IREE_HAL_QUEUE_AFFINITY_ANY, iree_hal_semaphore_list_empty(),
      iree_hal_fence_semaphore_list(alloca_fence.get()),
      IREE_HAL_ALLOCATOR_POOL_DEFAULT, output_buffer_params, output_byte_length,
      &output_buffer));

  // Wait for the alloca fence before executing cuDNN graph.
  IREE_RETURN_IF_ERROR(
      iree_hal_fence_wait(alloca_fence.get(), iree_infinite_timeout()));

  // Get underlying buffers from buffer view inputs.
  std::vector<iree_hal_buffer_t*> buffers(inputs.size() + 1);
  for (unsigned i = 0; i < inputs.size(); ++i) {
    buffers[i] = iree_hal_buffer_view_buffer(inputs[i].get());
  }
  buffers.back() = output_buffer.get();

  // TODO(ezhulenev): Allocate workspace required for running executable.
  IREE_RETURN_IF_ERROR(executable->Execute(handle_, buffers));

  // Wrap the buffer in a buffer view that provides the metadata for
  // runtime verification.
  vm::ref<iree_hal_buffer_view_t> output_view;
  std::vector<iree_host_size_t> output_shape = GetRowMajorShape(output);
  IREE_RETURN_IF_ERROR(iree_hal_buffer_view_create(
      output_buffer.get(), output_shape.size(), output_shape.data(),
      IREE_HAL_ELEMENT_TYPE_FLOAT_32, IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
      host_allocator_, &output_view));

  return output_view;
}

//===----------------------------------------------------------------------===//
// Functions dispatch table for CuDNNModuleState.
//===----------------------------------------------------------------------===//

using iree::vm::MakeNativeFunction;

using State = CuDNNModuleState;

static const vm::NativeFunction<State> kCuDNNModuleFunctions[] = {
    // Create cuDNN tensors
    MakeNativeFunction("tensor.create.4d", &State::TensorCreate<4>),
    MakeNativeFunction("tensor.create.4d.nhwc", &State::TensorCreate<4, NHWC>),

    // cuDNN operation graph construction
    MakeNativeFunction("operation_graph.create", &State::OperationGraphCreate),

    // cuDNN executable construction
    MakeNativeFunction("executable.create", &State::Executable),

    // Execute cuDNN executable with buffer inputs
    MakeNativeFunction("execute.2", &State::Execute<2>),
    MakeNativeFunction("execute.4", &State::Execute<4>),

    // cuDNN operations
    MakeNativeFunction("add", &State::Add),
    MakeNativeFunction("bias", &State::Bias),
    MakeNativeFunction("pointwise_relu", &State::PointwiseRelu),
    MakeNativeFunction("convolution.2d", &State::Convolution<2>),

    // Debugging operations
    MakeNativeFunction("debug.tensor", &State::PrintTensorDebug),
    MakeNativeFunction("debug.graph", &State::PrintGraphDebug),
};

//===----------------------------------------------------------------------===//
// CuDNN module instance that will be allocated and reused across contexts.
//===----------------------------------------------------------------------===//

class CuDNNModule final : public vm::NativeModule<CuDNNModuleState> {
 public:
  CuDNNModule(iree_vm_instance_t* instance, iree_hal_device_t* device,
              iree_allocator_t host_allocator, CUcontext cuda_ctx);

  StatusOr<std::unique_ptr<CuDNNModuleState>> CreateState(
      iree_allocator_t host_allocator) override;

 private:
  static constexpr uint32_t kVersion = 0;

  using NativeModule = vm::NativeModule<CuDNNModuleState>;

  // Retain a reference to the HAL (CUDA) device to keep CUDA context wrapper
  // alive for the duration of cuDNN module lifetime.
  vm::ref<iree_hal_device_t> device_;

  // CUDA context bound to the instance of a HAL CUDA device.
  CUcontext cuda_ctx_;
};

CuDNNModule::CuDNNModule(iree_vm_instance_t* instance,
                         iree_hal_device_t* device,
                         iree_allocator_t host_allocator, CUcontext cuda_ctx)
    : NativeModule("cudnn", CuDNNModule::kVersion, instance, host_allocator,
                   {kCuDNNModuleFunctions}),
      device_(vm::retain_ref(device)),
      cuda_ctx_(cuda_ctx) {}

StatusOr<std::unique_ptr<CuDNNModuleState>> CuDNNModule::CreateState(
    iree_allocator_t host_allocator) {
  // Load CUDA and resolbe API symbols.
  iree_hal_cuda_dynamic_symbols_t cuda_syms;
  IREE_RETURN_IF_ERROR(
      iree_hal_cuda_dynamic_symbols_initialize(host_allocator, &cuda_syms));

  // Load cuDNN library and resolve API symbols.
  openxla_cudnn_dynamic_symbols_t syms;
  IREE_RETURN_IF_ERROR(
      openxla_cudnn_dynamic_symbols_initialize(host_allocator, &syms));

  // Create a cuDNN handle for the new state object.
  cudnnHandle_t handle;
  // TODO: We must guarantee that `cuda_ctx_` is current when we create cuDNN
  // handle. Currently we rely on implicit guarantee that module is loaded
  // immediately after device is created, however it might not always be true?
  CUDNN_RETURN_IF_ERROR(&syms, cudnnCreate(&handle), "cudnnCreate");

  return std::make_unique<CuDNNModuleState>(device_.get(), host_allocator,
                                            cuda_syms, syms, handle);
}

}  // namespace openxla::runtime::nvgpu

//===----------------------------------------------------------------------===//
// Register cuDNN module with IREE runtime.
//===----------------------------------------------------------------------===//

using namespace openxla::runtime::nvgpu;

template <typename T>
static iree_status_t RegisterType(iree_vm_instance_t* instance,
                                  const char* type_name,
                                  iree_vm_ref_type_t* out_registration) {
  static iree_vm_ref_type_descriptor_t descriptor = {0};

  descriptor.type_name = iree_make_cstring_view(type_name);
  descriptor.offsetof_counter = T::offsetof_counter();
  descriptor.destroy = T::DirectDestroy;

  return iree_vm_instance_register_type(instance, &descriptor,
                                        out_registration);
}

extern "C" iree_status_t iree_custom_module_cudnn_create(
    iree_vm_instance_t* instance, iree_hal_device_t* device,
    iree_allocator_t host_allocator, iree_vm_module_t** out_module) {
  IREE_ASSERT_ARGUMENT(out_module);

  CUcontext cuda_ctx;
  IREE_RETURN_IF_ERROR(iree_hal_cuda_device_get_context(device, &cuda_ctx));
  auto module =
      std::make_unique<CuDNNModule>(instance, device, host_allocator, cuda_ctx);
  *out_module = module.release()->interface();

  return iree_ok_status();
}

extern "C" iree_status_t iree_custom_module_cudnn_register_types(
    iree_vm_instance_t* instance) {
  IREE_RETURN_IF_ERROR(RegisterType<CuDNNTensor>(instance, "cudnn.tensor",
                                                 &cudnn_tensor_registration));
  IREE_RETURN_IF_ERROR(RegisterType<CuDNNOperationGraph>(
      instance, "cudnn.operation_graph", &cudnn_operation_graph_registration));
  IREE_RETURN_IF_ERROR(RegisterType<CuDNNExecutable>(
      instance, "cudnn.executable", &cudnn_executable_registration));
  return iree_ok_status();
}
