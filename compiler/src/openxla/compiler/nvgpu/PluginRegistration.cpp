// Copyright 2023 The OpenXLA Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/PluginAPI/Client.h"
#include "mlir/Pass/Pass.h"
#include "openxla/compiler/nvgpu/Dialect/CUDNN/IR/CUDNNDialect.h"
#include "openxla/compiler/nvgpu/Dialect/CUDNN/Transforms/Passes.h"
#include "openxla/compiler/nvgpu/Transforms/Passes.h"

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

namespace {

namespace cudnn = openxla::compiler::nvgpu::cudnn;

struct NvgpuOptions {
  bool flag = false;

  void bindOptions(OptionsBinder &binder) {
    static llvm::cl::OptionCategory category("IREE NVGPU Plugin");
    binder.opt<bool>("openxla-nvgpu-flag", flag,
                     llvm::cl::desc("Dummy flag for the nvgpu plugin"),
                     llvm::cl::cat(category));
  }
};

struct NvgpuSession : public PluginSession<NvgpuSession, NvgpuOptions> {
  static void registerPasses() {
    ::detail::registerPasses();
    ::detail::cudnn::registerPasses();
  }

  void onRegisterDialects(DialectRegistry &registry) override {
    registry.insert<cudnn::CUDNNDialect>();
  }

  void extendInputConversionPreprocessingPassPipeline(
      OpPassManager &pm, InputDialectOptions::Type inputType) override {
    if (inputType == InputDialectOptions::Type::stablehlo) {
      pm.addPass(cudnn::createConvertHLOToCUDNNPass());
    }
  }

  void extendPreprocessingPassPipeline(OpPassManager &pm) override {
    pm.addPass(cudnn::createExpandCudnnOperationsPass());
    pm.addPass(cudnn::createConvertCudnnToRuntimePass());
  }
};

}  // namespace

IREE_DEFINE_COMPILER_OPTION_FLAGS(NvgpuOptions);

extern "C" bool iree_register_compiler_plugin_openxla_nvgpu(
    mlir::iree_compiler::PluginRegistrar *registrar) {
  registrar->registerPlugin<NvgpuSession>("openxla_nvgpu");
  return true;
}
