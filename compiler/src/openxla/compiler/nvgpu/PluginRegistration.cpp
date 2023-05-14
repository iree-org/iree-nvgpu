// Copyright 2023 The OpenXLA Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/PluginAPI/Client.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Pass/Pass.h"
#include "openxla/compiler/nvgpu/Dialect/CUDNN/IR/CUDNNDialect.h"
#include "openxla/compiler/nvgpu/Dialect/CUDNN/Transforms/Passes.h"
#include "openxla/compiler/nvgpu/Dialect/TritonFlow/IR/TritonFlowDialect.h"
#include "openxla/compiler/nvgpu/Dialect/TritonFlow/Transforms/Passes.h"
#include "openxla/compiler/nvgpu/Transforms/Passes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

using namespace mlir;
using namespace mlir::iree_compiler;

// TODO(ezhulenev): Move passes registration to `Passes.h`.
namespace detail {
namespace {
#define GEN_PASS_REGISTRATION
#include "openxla/compiler/nvgpu/Transforms/Passes.h.inc"
}  // namespace

namespace cudnn {
namespace {
#define GEN_PASS_REGISTRATION
#include "openxla/compiler/nvgpu/Dialect/CUDNN/Transforms/Passes.h.inc"
}  // namespace
}  // namespace cudnn

}  // namespace detail

//===----------------------------------------------------------------------===//
// OpenXLA compiler Triton plugin
//===----------------------------------------------------------------------===//

namespace {
using namespace ::openxla::compiler::nvgpu::triton;

struct TritonOptions {
  void bindOptions(OptionsBinder &binder) {}
};

struct TritonSession : public PluginSession<TritonSession, TritonOptions> {
  static void registerPasses() {
    registerOpenXlaTritonPases();
    registerOpenXlaTritonPipelines();
  }

  void onRegisterDialects(DialectRegistry &registry) override {
    registry.insert<TritonFlowDialect>();
    registry.insert<triton::TritonDialect>();
    registry.insert<NVVM::NVVMDialect>();
  }

  void extendPreprocessingPassPipeline(OpPassManager &pm) override {
    pm.addPass(createConvertTritonToFlowDispatchPass());
  }
};

}  // namespace

IREE_DEFINE_COMPILER_OPTION_FLAGS(TritonOptions);

//===----------------------------------------------------------------------===//
// OpenXLA compiler cuDNN  plugin
//===----------------------------------------------------------------------===//

namespace {
using namespace ::openxla::compiler::nvgpu::cudnn;

struct CudnnOptions {
  bool flag = false;

  void bindOptions(OptionsBinder &binder) {
    static llvm::cl::OptionCategory category("OpenXLA cuDNN Plugin");
    binder.opt<bool>("openxla-cudnn-flag", flag,
                     llvm::cl::desc("Dummy flag for the OpenXLA cuDNN plugin"),
  }
};

struct CudnnSession : public PluginSession<CudnnSession, CudnnOptions> {
  static void registerPasses() {
    ::detail::registerPasses();
    ::detail::cudnn::registerPasses();
  }

  void onRegisterDialects(DialectRegistry &registry) override {
    registry.insert<CUDNNDialect>();
  }

  void extendInputConversionPreprocessingPassPipeline(
      OpPassManager &pm, InputDialectOptions::Type inputType) override {
    if (inputType == InputDialectOptions::Type::stablehlo) {
      pm.addPass(createConvertHLOToCUDNNPass());
    }
  }

  void extendPreprocessingPassPipeline(OpPassManager &pm) override {
    pm.addPass(createExpandCudnnOperationsPass());
    pm.addPass(createConvertCudnnToRuntimePass());
  }
};

}  // namespace

IREE_DEFINE_COMPILER_OPTION_FLAGS(CudnnOptions);

//===----------------------------------------------------------------------===//
// OpenXLA compiler plugins registration
//===----------------------------------------------------------------------===//

extern "C" bool iree_register_compiler_plugin_openxla_nvgpu(
    mlir::iree_compiler::PluginRegistrar *registrar) {
  registrar->registerPlugin<CudnnSession>("openxla_nvgpu");
  registrar->registerPlugin<TritonSession>("openxla-triton");
  return true;
}
