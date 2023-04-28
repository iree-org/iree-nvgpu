// Copyright 2023 The OpenXLA Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef OPENXLA_RUNTIME_NVGPU_CUDNN_API_H_
#define OPENXLA_RUNTIME_NVGPU_CUDNN_API_H_

#define NV_CUDNN_DISABLE_EXCEPTION

#include <cudnn_frontend.h>
#include <iree/hal/buffer.h>
#include <iree/vm/ref_cc.h>

#include <optional>

#include "iree/base/internal/span.h"
#include "iree/vm/api.h"
#include "openxla/runtime/nvgpu/dynamic_symbols.h"

namespace openxla::runtime::nvgpu {

// OpenXLA wrappers around cudnn_frontend primitives to simplify operation graph
// construction and execution. We rely on cudnn_frontend to provide C++ RAII
// wrappers for all cuDNN Graph API backend descriptors.

//===----------------------------------------------------------------------===//
// cuDNN tensor representing an abstract shaped and typed block of memory
//===----------------------------------------------------------------------===//

class CudnnTensor : public iree::vm::RefObject<CudnnTensor> {
 public:
  enum class Kind { kArg, kOpResult };

  explicit CudnnTensor(Kind kind) : kind_(kind) {}
  virtual ~CudnnTensor() = default;

  virtual const cudnn_frontend::Tensor& tensor() const = 0;

  const cudnn_frontend::Tensor* operator->() { return &tensor(); }

  Kind kind() const { return kind_; }

 private:
  Kind kind_;
};

//===----------------------------------------------------------------------===//
// Tensor corresponding to the cuDNN graph arguments
//===----------------------------------------------------------------------===//

class CudnnArgTensor final : public CudnnTensor {
 public:
  CudnnArgTensor(openxla_cudnn_dynamic_symbols_t* syms,
                 cudnn_frontend::Tensor tensor);
  ~CudnnArgTensor() override;

  const cudnn_frontend::Tensor& tensor() const override;

  static bool classof(const CudnnTensor* tensor) {
    return tensor->kind() == Kind::kArg;
  }

 private:
  openxla_cudnn_dynamic_symbols_t* syms_;
  std::optional<cudnn_frontend::Tensor> tensor_;
};

//===----------------------------------------------------------------------===//
// Tensor corresponding to the cuDNN operation result
//===----------------------------------------------------------------------===//

class CudnnOpResultTensor final : public CudnnTensor {
 public:
  CudnnOpResultTensor(openxla_cudnn_dynamic_symbols_t* syms,
                      iree::span<CudnnTensor* const> inputs,
                      cudnn_frontend::Operation operation,
                      cudnn_frontend::Tensor tensor);
  ~CudnnOpResultTensor() override;

  std::vector<CudnnTensor*> inputs() const;
  const cudnn_frontend::Operation* operation() const;
  const cudnn_frontend::Tensor& tensor() const override;

  static bool classof(const CudnnTensor* tensor) {
    return tensor->kind() == Kind::kOpResult;
  }

 private:
  openxla_cudnn_dynamic_symbols_t* syms_;

  // Tensors inputs to the cuDNN operation. We need to keep a reference to them
  // to be able to traverse use-def chains when building an operation graph.
  std::vector<iree::vm::ref<CudnnTensor>> inputs_;

  // cuDNN operation that computes the result tensor.
  std::optional<cudnn_frontend::Operation> operation_;
  std::optional<cudnn_frontend::Tensor> tensor_;
};

//===----------------------------------------------------------------------===//
// Cudnn operation graph
//===----------------------------------------------------------------------===//

class CudnnOperationGraph : public iree::vm::RefObject<CudnnOperationGraph> {
 public:
  CudnnOperationGraph(openxla_cudnn_dynamic_symbols_t* syms,
                      cudnn_frontend::OperationGraph graph,
                      iree::span<CudnnTensor* const> args,
                      iree::span<CudnnTensor* const> rets);
  ~CudnnOperationGraph();

  cudnn_frontend::OperationGraph& graph();

  std::vector<CudnnTensor*> args() const;
  std::vector<CudnnTensor*> rets() const;

  iree::span<const int64_t> uids() const;

 private:
  openxla_cudnn_dynamic_symbols_t* syms_;
  std::optional<cudnn_frontend::OperationGraph> graph_;

  std::vector<iree::vm::ref<CudnnTensor>> args_;
  std::vector<iree::vm::ref<CudnnTensor>> rets_;

  // Ids of tensors in `args_` and `rets_`.
  std::vector<int64_t> uids_;
};

//===----------------------------------------------------------------------===//
// Cudnn executable
//===----------------------------------------------------------------------===//

// Cudnn executable encapsulates all the details of configuring cuDNN engines
// and execution plans for the given operation graph: filtering available engine
// configs, selecting the best engine config using cuDNN heuristics, auto-tuning
// at run time to find the best-performing config.
class CudnnExecutable : public iree::vm::RefObject<CudnnExecutable> {
 public:
  CudnnExecutable(openxla_cudnn_dynamic_symbols_t* syms,
                  CudnnOperationGraph& graph,
                  iree::span<const cudnn_frontend::ExecutionPlan> plans);
  ~CudnnExecutable();

  const CudnnOperationGraph& graph() const;

  // Executes operation graph with user provided buffers using one of the
  // available execution plans.
  iree::Status Execute(cudnnHandle_t handle,
                       iree::span<iree_hal_buffer_t* const> buffers);

  // TODO(ezhulenev): Add functions for auto-tuning executable to pick the best
  // performing execution plan at run time.

 private:
  openxla_cudnn_dynamic_symbols_t* syms_;
  iree::vm::ref<CudnnOperationGraph> graph_;
  std::vector<cudnn_frontend::ExecutionPlan> plans_;
};

//===----------------------------------------------------------------------===//
// Wrappers around cuDNN APIs export from a cuDNN module to the user
//===----------------------------------------------------------------------===//

// Creates a tensor placeholder for cuDNN graph argument.
iree::StatusOr<iree::vm::ref<CudnnTensor>> CreateTensor(
    openxla_cudnn_dynamic_symbols_t* syms, iree::span<const int64_t> dims,
    iree::span<const int64_t> strides, int64_t uid, cudnnDataType_t dtype,
    int64_t alignment);

// Creates a pointwise relu operation.
iree::StatusOr<iree::vm::ref<CudnnTensor>> CreatePointwiseRelu(
    openxla_cudnn_dynamic_symbols_t* syms, CudnnTensor& input,
    double lower_clip, double upper_clip, int64_t uid, int64_t alignment,
    bool is_virtual);

// Creates a pointwise unary operation.
iree::StatusOr<iree::vm::ref<CudnnTensor>> CreatePointwiseUnary(
    openxla_cudnn_dynamic_symbols_t* syms, cudnnPointwiseMode_t mode,
    CudnnTensor& x, float alpha, int64_t uid, int64_t alignment,
    bool is_virtual);

// Creates a pointwise binary operation.
iree::StatusOr<iree::vm::ref<CudnnTensor>> CreatePointwiseBinary(
    openxla_cudnn_dynamic_symbols_t* syms, cudnnPointwiseMode_t mode,
    CudnnTensor& x, float alpha, CudnnTensor& b, float alpha2, int64_t uid,
    int64_t alignment, bool is_virtual);

// Creates a forward convolution operation.
iree::StatusOr<iree::vm::ref<CudnnTensor>> CreateConvolution(
    openxla_cudnn_dynamic_symbols_t* syms, CudnnTensor& input,
    CudnnTensor& filter, int64_t uid, int64_t alignment, bool is_virtual,
    cudnnConvolutionMode_t mode);

// Creates an operation graph computing tensor results.
iree::StatusOr<iree::vm::ref<CudnnOperationGraph>> CreateOperationGraph(
    openxla_cudnn_dynamic_symbols_t* syms, cudnnHandle_t handle,
    iree::span<CudnnTensor* const> rets);

// Creates an executable from the operation graph.
iree::StatusOr<iree::vm::ref<CudnnExecutable>> CreateExecutable(
    openxla_cudnn_dynamic_symbols_t* syms, cudnnHandle_t handle,
    CudnnOperationGraph& graph);

//===----------------------------------------------------------------------===//
// Helper functions for setting up cuDNN descriptors
//===----------------------------------------------------------------------===//

// Get strides for row major storage format (in cuDNN NCHW, NCDHW are considered
// to be the row-major formats for 4-D and 5-D tensors).
std::vector<int64_t> GetRowMajorStrides(iree::span<const int64_t> dims);

// Compute strides for channels last storage format (NHWC, NDHWC). Can be
// computed only for 4-D or 5-D tensors.
std::vector<int64_t> GetChannelsLastStrides(iree::span<const int64_t> dims);

}  // namespace openxla::runtime::nvgpu

//===----------------------------------------------------------------------===//
// Register types with IREE VM
//===----------------------------------------------------------------------===//

IREE_VM_DECLARE_TYPE_ADAPTERS(cudnn_tensor,
                              openxla::runtime::nvgpu::CudnnTensor);
IREE_VM_DECLARE_TYPE_ADAPTERS(cudnn_operation_graph,
                              openxla::runtime::nvgpu::CudnnOperationGraph);
IREE_VM_DECLARE_TYPE_ADAPTERS(cudnn_executable,
                              openxla::runtime::nvgpu::CudnnExecutable);

#endif  // OPENXLA_RUNTIME_NVGPU_CUDNN_API_H_
