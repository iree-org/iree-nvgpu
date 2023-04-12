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
// CuDNN module state encapsulates all the state required for running cuDNN
// operations (launching cuDNN graphs on a stream) at run time.
//===----------------------------------------------------------------------===//

class CuDNNModuleState {
 public:
  CuDNNModuleState(iree_hal_device_t* device, iree_allocator_t host_allocator,
                   iree_hal_cuda_dynamic_symbols_t cuda_syms,
                   openxla_cudnn_dynamic_symbols_t syms, cudnnHandle_t handle);
  ~CuDNNModuleState();

  enum class TensorFormat { kRowMajor, kChannelsLast };

  // Creates a new tensor for cuDNN graph argument.
  template <TensorFormat format = TensorFormat::kRowMajor>
  StatusOr<vm::ref<CuDNNTensor>> Argument(int64_t dtype,
                                          const vm::ref<iree_vm_list_t> dims,
                                          int64_t uid, int64_t alignment);

  // Creates a pointwise relu operation and returns result tensor.
  StatusOr<vm::ref<CuDNNTensor>> PointwiseRelu(const vm::ref<CuDNNTensor> input,
                                               float lower_clip,
                                               float upper_clip, int64_t uid,
                                               int64_t alignment);

  // Creates a convolution operation and returns result tensor.
  StatusOr<vm::ref<CuDNNTensor>> Convolution(const vm::ref<CuDNNTensor> input,
                                             const vm::ref<CuDNNTensor> filter,
                                             int64_t uid, int64_t alignment);

  // TODO(ezhulenev): To be able to pass a list of tensors, `!cudnn.tensor` has
  // to be registered as a ref type (see `IREE::VM::RefType` and`!vmvx.buffer`
  // which is registered as reference type and can be added to the list).

  // Creates a cuDNN graph computing `tensor` result.
  StatusOr<vm::ref<CuDNNOperationGraph>> CreateGraph(
      const vm::ref<CuDNNTensor> tensor);

  // Creates a cuDNN executable from the given operation graph.
  StatusOr<vm::ref<CuDNNExecutable>> Executable(
      const vm::ref<CuDNNOperationGraph> graph);

  // TODO(ezhulenev): This is proof of concept for executing a cuDNN graph with
  // a single convolution operation. We need to figure out how to pass a list
  // of buffers to graph inputs, and potentially return multiple results.

  // Executes cuDNN executable with given HAL buffer view inputs and returns
  // result as a HAL buffer view.
  StatusOr<vm::ref<iree_hal_buffer_view_t>> Execute(
      const vm::ref<CuDNNExecutable> executable,
      const vm::ref<iree_hal_buffer_view_t> input,
      const vm::ref<iree_hal_buffer_view_t> filter,
      const vm::ref<iree_hal_fence_t> wait_fence,
      const vm::ref<iree_hal_fence_t> signal_fence);

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

static StatusOr<std::vector<int64_t>> LoadI64Vec(const iree_vm_list_t* list) {
  std::vector<int64_t> vector(iree_vm_list_size(list));
  for (size_t i = 0; i < vector.size(); ++i) {
    iree_vm_value_t value;
    IREE_RETURN_IF_ERROR(
        iree_vm_list_get_value_as(list, i, IREE_VM_VALUE_TYPE_I64, &value));
    vector[i] = value.i64;
  }
  return vector;
}

template <CuDNNModuleState::TensorFormat format>
StatusOr<vm::ref<CuDNNTensor>> CuDNNModuleState::Argument(
    int64_t dtype, const vm::ref<iree_vm_list_t> dims, int64_t uid,
    int64_t alignment) {
  IREE_ASSIGN_OR_RETURN(cudnnDataType_t data_type, ToCudnnDataType(dtype));
  IREE_ASSIGN_OR_RETURN(std::vector<int64_t> dimensions, LoadI64Vec(&*dims));

  std::vector<int64_t> strides = format == TensorFormat::kRowMajor
                                     ? GetRowMajorStrides(dimensions)
                                     : GetChannelsLastStrides(dimensions);
  return CreateArgument(&syms_, dimensions, strides, uid, data_type, alignment);
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
    int64_t uid, int64_t alignment) {
  return CreatePointwiseRelu(&syms_, *input, lower_clip, upper_clip, uid,
                             alignment);
}

StatusOr<vm::ref<CuDNNTensor>> CuDNNModuleState::Convolution(
    const vm::ref<CuDNNTensor> input, const vm::ref<CuDNNTensor> filter,
    int64_t uid, int64_t alignment) {
  return CreateConvolution(&syms_, *input, *filter, uid, alignment);
}

StatusOr<vm::ref<CuDNNOperationGraph>> CuDNNModuleState::CreateGraph(
    const vm::ref<CuDNNTensor> tensor) {
  return CreateOperationGraph(&syms_, handle_, {tensor.get()});
}

StatusOr<vm::ref<CuDNNExecutable>> CuDNNModuleState::Executable(
    const vm::ref<CuDNNOperationGraph> graph) {
  return CreateExecutable(&syms_, handle_, *graph);
}

StatusOr<vm::ref<iree_hal_buffer_view_t>> CuDNNModuleState::Execute(
    const vm::ref<CuDNNExecutable> executable,
    const vm::ref<iree_hal_buffer_view_t> input,
    const vm::ref<iree_hal_buffer_view_t> filter,
    const vm::ref<iree_hal_fence_t> wait_fence,
    const vm::ref<iree_hal_fence_t> signal_fence) {
  // Arguments and results defined by the operation graph.
  std::vector<CuDNNTensor*> args = executable->graph().args();
  std::vector<CuDNNTensor*> rets = executable->graph().rets();

  // TODO(ezhulenev): Remove this asserts once we support more complex graphs.
  IREE_ASSERT_EQ(args.size(), 2);
  IREE_ASSERT_EQ(rets.size(), 1);

  // Tensors required for running single convolution operation.
  // const cudnn_frontend::Tensor& input = args[0]->tensor();
  // const cudnn_frontend::Tensor& filter = args[1]->tensor();
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
      device_, IREE_HAL_QUEUE_AFFINITY_ANY,
      iree_hal_fence_semaphore_list(wait_fence.get()),
      iree_hal_fence_semaphore_list(alloca_fence.get()),
      IREE_HAL_ALLOCATOR_POOL_DEFAULT, output_buffer_params, output_byte_length,
      &output_buffer));

  // Wait for the alloca fence before executing cuDNN graph.
  IREE_RETURN_IF_ERROR(
      iree_hal_fence_wait(alloca_fence.get(), iree_infinite_timeout()));

  std::vector<iree_hal_buffer_t*> buffers = {
      iree_hal_buffer_view_buffer(input.get()),
      iree_hal_buffer_view_buffer(filter.get()), output_buffer.get()};

  // TODO(ezhulenev): Allocate workspace required for running executable.
  IREE_RETURN_IF_ERROR(executable->Execute(handle_, buffers));

  // Signal fence after completing execution.
  IREE_RETURN_IF_ERROR(iree_hal_fence_signal(signal_fence.get()));

  // Wrap the buffer in a buffer view that provides the metadata for
  // runtime verification.
  vm::ref<iree_hal_buffer_view_t> output_view;
  std::vector<iree_host_size_t> dims(output.getDim(),
                                     output.getDim() + output.getDimCount());
  IREE_RETURN_IF_ERROR(iree_hal_buffer_view_create(
      output_buffer.get(), dims.size(), dims.data(),
      IREE_HAL_ELEMENT_TYPE_FLOAT_32, IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
      host_allocator_, &output_view));

  return output_view;
}

//===----------------------------------------------------------------------===//
// Functions dispatch table for CuDNNModuleState.
//===----------------------------------------------------------------------===//

using iree::vm::MakeNativeFunction;

static constexpr CuDNNModuleState::TensorFormat NHWC =
    CuDNNModuleState::TensorFormat::kChannelsLast;

static const vm::NativeFunction<CuDNNModuleState> kCuDNNModuleFunctions[] = {
    // Tensor arguments
    MakeNativeFunction("tensor.arg", &CuDNNModuleState::Argument<>),
    MakeNativeFunction("tensor.arg.nhwc", &CuDNNModuleState::Argument<NHWC>),

    // cuDNN operations
    MakeNativeFunction("pointwise_relu", &CuDNNModuleState::PointwiseRelu),
    MakeNativeFunction("convolution", &CuDNNModuleState::Convolution),

    // cuDNN graph construction
    MakeNativeFunction("graph.create", &CuDNNModuleState::CreateGraph),

    // cuDNN executable construction
    MakeNativeFunction("executable.create", &CuDNNModuleState::Executable),

    // Execute cuDNN executable with buffer inputs
    MakeNativeFunction("execute", &CuDNNModuleState::Execute),

    // Debugging operations
    MakeNativeFunction("debug.tensor", &CuDNNModuleState::PrintTensorDebug),
    MakeNativeFunction("debug.graph", &CuDNNModuleState::PrintGraphDebug),
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
