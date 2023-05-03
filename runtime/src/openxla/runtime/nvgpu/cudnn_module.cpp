// Copyright 2023 The OpenXLA Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "openxla/runtime/nvgpu/cudnn_module.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>

#include "iree/hal/drivers/cuda/dynamic_symbols.h"
#include "iree/modules/hal/types.h"
#include "iree/vm/dynamic/api.h"
#include "iree/vm/native_module_cc.h"
#include "iree/vm/ref_cc.h"
#include "openxla/runtime/nvgpu/cudnn_api.h"
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

using KHWC = NHWC;

//===----------------------------------------------------------------------===//
// Cudnn module state encapsulates all the state required for running cuDNN
// operations (launching cuDNN graphs on a stream) at run time
//===----------------------------------------------------------------------===//

class CudnnModuleState {
  // TODO(ezhulenev): This is a random value and never really verified at run
  // time. We always have to check that buffers we see at run time are properly
  // aligned according to what we promised to cuDNN.
  static constexpr int64_t kAlignment = 32;

 public:
  CudnnModuleState(iree_allocator_t host_allocator,
                   iree_hal_cuda_dynamic_symbols_t* cuda_syms,
                   openxla_cudnn_dynamic_symbols_t* syms);
  ~CudnnModuleState();

  enum class TensorFormat { kRowMajor, kChannelsLast };

  // Creates a new tensor with given shape and layout.
  template <size_t rank, typename Layout = RowMajor>
  StatusOr<vm::ref<CudnnTensor>> TensorCreate(
      int64_t dtype, std::array<int64_t, rank> dimensions);

  // Creates a cuDNN module handle from a HAL device.
  StatusOr<vm::ref<CudnnHandle>> Handle(
      const vm::ref<iree_hal_device_t> device);

  // Creates a cuDNN operation graph computing `tensor` result.
  StatusOr<vm::ref<CudnnOperationGraph>> OperationGraphCreate(
      const vm::ref<CudnnHandle> handle, const vm::ref<CudnnTensor> tensor);

  // Creates a cuDNN executable from the given operation graph.
  StatusOr<vm::ref<CudnnExecutable>> Executable(
      const vm::ref<CudnnOperationGraph> graph);

  // Executes cuDNN executable with given HAL buffer view inputs and returns
  // result as a HAL buffer view.
  template <size_t n>
  StatusOr<vm::ref<iree_hal_buffer_view_t>> Execute(
      const vm::ref<CudnnExecutable> executable,
      std::array<vm::ref<iree_hal_buffer_view_t>, n> inputs);

  // Creates a pointwise relu operation and returns result tensor.
  StatusOr<vm::ref<CudnnTensor>> Relu(const vm::ref<CudnnTensor> input,
                                      float lower_clip, float upper_clip,
                                      int64_t uid, int64_t alignment,
                                      int32_t is_virtual);

  // Creates a pointwise unary operation and returns a result tensor.
  template <cudnnPointwiseMode_t mode>
  StatusOr<vm::ref<CudnnTensor>> PointwiseUnary(const vm::ref<CudnnTensor> x,
                                                float alpha,
                                                int32_t is_virtual);

  // Creates a pointwise binary operation and returns a result tensor.
  template <cudnnPointwiseMode_t mode>
  StatusOr<vm::ref<CudnnTensor>> PointwiseBinary(const vm::ref<CudnnTensor> x,
                                                 float alpha,
                                                 const vm::ref<CudnnTensor> b,
                                                 float alpha2,
                                                 int32_t is_virtual);

  // Creates a bias operation and returns a result tensor.
  StatusOr<vm::ref<CudnnTensor>> Bias(const vm::ref<CudnnTensor> x,
                                      const vm::ref<CudnnTensor> b,
                                      int32_t is_virtual);

  // Creates a convolution operation and returns result tensor.
  template <size_t spatial_dims>
  StatusOr<vm::ref<CudnnTensor>> Convolution(
      const vm::ref<CudnnTensor> x, const vm::ref<CudnnTensor> w,
      std::array<int64_t, spatial_dims> stride,
      std::array<int64_t, spatial_dims> pre_padding,
      std::array<int64_t, spatial_dims> post_padding,
      std::array<int64_t, spatial_dims> dilation, int32_t is_virtual,
      int32_t mode);

  // Prints tensor debug information to stderr.
  Status PrintTensorDebug(const vm::ref<CudnnTensor> tensor);

  // Prints graph debug information to stderr.
  Status PrintGraphDebug(const vm::ref<CudnnOperationGraph> graph);

 private:
  CudnnModuleState(const CudnnModuleState&) = delete;
  CudnnModuleState& operator=(const CudnnModuleState&) = delete;

  iree_allocator_t host_allocator_;

  iree_hal_cuda_dynamic_symbols_t* cuda_syms_;
  openxla_cudnn_dynamic_symbols_t* syms_;

  // We use automatic uid assignment for all cuDNN tensors in the graph.
  uint64_t uid_ = 0;
};

CudnnModuleState::CudnnModuleState(iree_allocator_t host_allocator,
                                   iree_hal_cuda_dynamic_symbols_t* cuda_syms,
                                   openxla_cudnn_dynamic_symbols_t* syms)
    : host_allocator_(host_allocator), cuda_syms_(cuda_syms), syms_(syms) {}

CudnnModuleState::~CudnnModuleState() {}

static StatusOr<cudnnDataType_t> ToCudnnDataType(int64_t dtype) {
  if (dtype < CUDNN_DATA_FLOAT || dtype > CUDNN_DATA_BOOLEAN)
    return Status(StatusCode::kInvalidArgument, "unsupported data type");
  return static_cast<cudnnDataType_t>(dtype);
}

static StatusOr<cudnnConvolutionMode_t> ToCudnnConvolutionMode(int32_t mode) {
  if (mode < CUDNN_CONVOLUTION || mode > CUDNN_CROSS_CORRELATION)
    return Status(StatusCode::kInvalidArgument, "unsupported convolution mode");
  return static_cast<cudnnConvolutionMode_t>(mode);
}

template <size_t rank, typename Layout>
StatusOr<vm::ref<CudnnTensor>> CudnnModuleState::TensorCreate(
    int64_t dtype, std::array<int64_t, rank> dimensions) {
  IREE_ASSIGN_OR_RETURN(cudnnDataType_t data_type, ToCudnnDataType(dtype));
  std::array<int64_t, rank> strides = Layout::strides(dimensions);
  return CreateTensor(syms_, dimensions, strides, uid_++, data_type,
                      kAlignment);
}

Status CudnnModuleState::PrintTensorDebug(const vm::ref<CudnnTensor> tensor) {
  std::string desc = tensor->tensor().describe();
  fprintf(stderr, "Tensor: %s\n", desc.c_str());
  return OkStatus();
}

Status CudnnModuleState::PrintGraphDebug(
    const vm::ref<CudnnOperationGraph> graph) {
  std::string desc = graph->graph().describe();
  fprintf(stderr, "Graph: %s\n", desc.c_str());
  return OkStatus();
}

StatusOr<vm::ref<CudnnTensor>> CudnnModuleState::Relu(
    const vm::ref<CudnnTensor> input, float lower_clip, float upper_clip,
    int64_t uid, int64_t alignment, int32_t is_virtual) {
  return CreateRelu(syms_, *input, lower_clip, upper_clip, uid, alignment,
                    is_virtual);
}

template <cudnnPointwiseMode_t mode>
StatusOr<vm::ref<CudnnTensor>> CudnnModuleState::PointwiseUnary(
    const vm::ref<CudnnTensor> x, float alpha, int32_t is_virtual) {
  return CreatePointwiseUnary(syms_, mode, *x, alpha, uid_++, kAlignment,
                              is_virtual);
}

template <cudnnPointwiseMode_t mode>
StatusOr<vm::ref<CudnnTensor>> CudnnModuleState::PointwiseBinary(
    const vm::ref<CudnnTensor> x, float alpha, const vm::ref<CudnnTensor> b,
    float alpha2, int32_t is_virtual) {
  return CreatePointwiseBinary(syms_, mode, *x, alpha, *b, alpha2, uid_++,
                               kAlignment, is_virtual);
}

StatusOr<vm::ref<CudnnTensor>> CudnnModuleState::Bias(
    const vm::ref<CudnnTensor> x, const vm::ref<CudnnTensor> b,
    int32_t is_virtual) {
  return CreatePointwiseBinary(syms_, CUDNN_POINTWISE_ADD, *x, 1.0, *b, 1.0,
                               uid_++, kAlignment, is_virtual);
}

template <size_t spatial_dims>
StatusOr<vm::ref<CudnnTensor>> CudnnModuleState::Convolution(
    const vm::ref<CudnnTensor> x, const vm::ref<CudnnTensor> w,
    std::array<int64_t, spatial_dims> stride,
    std::array<int64_t, spatial_dims> pre_padding,
    std::array<int64_t, spatial_dims> post_padding,
    std::array<int64_t, spatial_dims> dilation, int32_t is_virtual,
    int32_t mode) {
  IREE_ASSIGN_OR_RETURN(cudnnConvolutionMode_t conv_mode,
                        ToCudnnConvolutionMode(mode));
  return CreateConvolution(syms_, *x, *w, stride, pre_padding, post_padding,
                           dilation, uid_++, kAlignment, is_virtual, conv_mode);
}

StatusOr<vm::ref<CudnnHandle>> CudnnModuleState::Handle(
    const vm::ref<iree_hal_device_t> device) {
  return CreateHandle(syms_, device.get());
}

StatusOr<vm::ref<CudnnOperationGraph>> CudnnModuleState::OperationGraphCreate(
    const vm::ref<CudnnHandle> handle, const vm::ref<CudnnTensor> tensor) {
  return CreateOperationGraph(syms_, *handle, {tensor.get()});
}

StatusOr<vm::ref<CudnnExecutable>> CudnnModuleState::Executable(
    const vm::ref<CudnnOperationGraph> graph) {
  return CreateExecutable(cuda_syms_, syms_, *graph);
}

template <size_t n>
StatusOr<vm::ref<iree_hal_buffer_view_t>> CudnnModuleState::Execute(
    const vm::ref<CudnnExecutable> executable,
    std::array<vm::ref<iree_hal_buffer_view_t>, n> inputs) {
  std::array<iree_hal_buffer_view_t*, n> args;
  for (size_t i = 0; i < n; ++i) args[i] = inputs[i].get();

  IREE_ASSIGN_OR_RETURN(vm::ref<iree_hal_buffer_view_t> result,
                        executable->Execute(host_allocator_, args));

  return result;
}

//===----------------------------------------------------------------------===//
// Functions dispatch table for CudnnModuleState
//===----------------------------------------------------------------------===//

using iree::vm::MakeNativeFunction;

using State = CudnnModuleState;

static const vm::NativeFunction<State> kCudnnModuleFunctions[] = {
    // Create cuDNN tensors
    MakeNativeFunction("tensor.create.4d", &State::TensorCreate<4>),
    MakeNativeFunction("tensor.create.4d.nhwc", &State::TensorCreate<4, NHWC>),
    MakeNativeFunction("tensor.create.4d.khwc", &State::TensorCreate<4, KHWC>),

    // cuDNN handle operations
    MakeNativeFunction("handle", &State::Handle),

    // cuDNN operation graph construction
    MakeNativeFunction("operation_graph.create", &State::OperationGraphCreate),

    // cuDNN executable construction
    MakeNativeFunction("executable.create", &State::Executable),

    // Execute cuDNN executable with buffer inputs
    MakeNativeFunction("execute.1", &State::Execute<1>),
    MakeNativeFunction("execute.2", &State::Execute<2>),
    MakeNativeFunction("execute.3", &State::Execute<3>),
    MakeNativeFunction("execute.4", &State::Execute<4>),
    MakeNativeFunction("execute.5", &State::Execute<5>),
    MakeNativeFunction("execute.6", &State::Execute<6>),
    MakeNativeFunction("execute.7", &State::Execute<7>),
    MakeNativeFunction("execute.8", &State::Execute<8>),

    // cuDNN pointwise unary operations
    MakeNativeFunction("sqrt", &State::PointwiseUnary<CUDNN_POINTWISE_SQRT>),

    // cuDNN pointwise binary operations
    MakeNativeFunction("add", &State::PointwiseBinary<CUDNN_POINTWISE_ADD>),
    MakeNativeFunction("div", &State::PointwiseBinary<CUDNN_POINTWISE_DIV>),
    MakeNativeFunction("sub", &State::PointwiseBinary<CUDNN_POINTWISE_SUB>),
    MakeNativeFunction("max", &State::PointwiseBinary<CUDNN_POINTWISE_MAX>),
    MakeNativeFunction("mul", &State::PointwiseBinary<CUDNN_POINTWISE_MUL>),

    // cuDNN operations
    MakeNativeFunction("bias", &State::Bias),
    MakeNativeFunction("relu", &State::Relu),
    MakeNativeFunction("convolution.2d", &State::Convolution<2>),

    // Debugging operations
    MakeNativeFunction("debug.tensor", &State::PrintTensorDebug),
    MakeNativeFunction("debug.graph", &State::PrintGraphDebug),
};

//===----------------------------------------------------------------------===//
// Cudnn module instance that will be allocated and reused across contexts
//===----------------------------------------------------------------------===//

class CudnnModule final : public vm::NativeModule<CudnnModuleState> {
 public:
  CudnnModule(iree_vm_instance_t* instance, iree_allocator_t host_allocator);

  StatusOr<std::unique_ptr<CudnnModuleState>> CreateState(
      iree_allocator_t host_allocator) override;

 private:
  static constexpr uint32_t kVersion = 0;

  using NativeModule = vm::NativeModule<CudnnModuleState>;

  iree_hal_cuda_dynamic_symbols_t cuda_syms_;
  iree_status_t cuda_syms_status_;

  openxla_cudnn_dynamic_symbols_t syms_;
  iree_status_t syms_status_;
};

CudnnModule::CudnnModule(iree_vm_instance_t* instance,
                         iree_allocator_t host_allocator)
    : NativeModule("cudnn", CudnnModule::kVersion, instance, host_allocator,
                   {kCudnnModuleFunctions}) {
  // Load CUDA and cuDNN libraries and resolve API symbols.
  cuda_syms_status_ =
      iree_hal_cuda_dynamic_symbols_initialize(host_allocator, &cuda_syms_);
  syms_status_ =
      openxla_cudnn_dynamic_symbols_initialize(host_allocator, &syms_);
}

StatusOr<std::unique_ptr<CudnnModuleState>> CudnnModule::CreateState(
    iree_allocator_t host_allocator) {
  IREE_RETURN_IF_ERROR(cuda_syms_status_);
  IREE_RETURN_IF_ERROR(syms_status_);
  return std::make_unique<CudnnModuleState>(host_allocator, &cuda_syms_,
                                            &syms_);
}

//===----------------------------------------------------------------------===//
// Register cuDNN module with IREE runtime
//===----------------------------------------------------------------------===//

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

}  // namespace openxla::runtime::nvgpu

//===----------------------------------------------------------------------===//
// Static IREE VM module registration
//===----------------------------------------------------------------------===//

using namespace openxla::runtime::nvgpu;

extern "C" iree_status_t openxla_nvgpu_cudnn_module_create(
    iree_vm_instance_t* instance, iree_allocator_t host_allocator,
    iree_vm_module_t** out_module) {
  IREE_ASSERT_ARGUMENT(out_module);

  auto module = std::make_unique<CudnnModule>(instance, host_allocator);
  *out_module = module.release()->interface();

  return iree_ok_status();
}

extern "C" iree_status_t openxla_nvgpu_cudnn_module_register_types(
    iree_vm_instance_t* instance) {
  IREE_RETURN_IF_ERROR(RegisterType<CudnnTensor>(instance, "cudnn.tensor",
                                                 &cudnn_tensor_registration));
  IREE_RETURN_IF_ERROR(RegisterType<CudnnHandle>(instance, "cudnn.handle",
                                                 &cudnn_handle_registration));
  IREE_RETURN_IF_ERROR(RegisterType<CudnnOperationGraph>(
      instance, "cudnn.operation_graph", &cudnn_operation_graph_registration));
  IREE_RETURN_IF_ERROR(RegisterType<CudnnExecutable>(
      instance, "cudnn.executable", &cudnn_executable_registration));
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Dynamic IREE VM module registration
//===----------------------------------------------------------------------===//

extern "C" IREE_VM_DYNAMIC_MODULE_EXPORT iree_status_t
openxla_create_cudnn_module(iree_vm_dynamic_module_version_t max_version,
                            iree_vm_instance_t* instance,
                            iree_host_size_t param_count,
                            const iree_string_pair_t* params,
                            iree_allocator_t host_allocator,
                            iree_vm_module_t** out_module) {
  // Ensure the version matches; the version will change if the VM module
  // interface changes and existing libraries are incompatible.
  if (max_version != IREE_VM_DYNAMIC_MODULE_VERSION_LATEST) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "unsupported runtime version %u, module compiled with version %u",
        max_version, IREE_VM_DYNAMIC_MODULE_VERSION_LATEST);
  }

#if IREE_TRACING_FEATURES
  // Today Tracy cannot be used with custom dynamic modules as it'll try to
  // create a new tracing context distinct from the hosting application. Custom
  // module libraries should be built with tracing disabled.
  fprintf(stderr,
          "Tracy is not currently supported in custom dynamic modules\n");
#endif  // IREE_TRACING_FEATURES

  // Ensure HAL types are available. We need to do this as we're being
  // dynamically loaded and can't automatically access the hosting process
  // variables.
  IREE_RETURN_IF_ERROR(iree_hal_module_resolve_all_types(instance));
  IREE_RETURN_IF_ERROR(openxla_nvgpu_cudnn_module_register_types(instance));
  return openxla_nvgpu_cudnn_module_create(instance, host_allocator,
                                           out_module);
}
