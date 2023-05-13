// Copyright 2023 The OpenXLA Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "openxla/compiler/nvgpu/Dialect/Triton/Transforms/Passes.h"

#include "openxla/compiler/nvgpu/Dialect/Triton/Conversion/ConvertTritonToCustomDispatch.h"

namespace detail {
namespace {
#define GEN_PASS_REGISTRATION
#include "openxla/compiler/nvgpu/Dialect/Triton/Transforms/Passes.h.inc"
}  // namespace
}  // namespace detail

namespace openxla::compiler::nvgpu::triton {
using namespace mlir;

void registerOpenXlaTritonPases() { ::detail::registerPasses(); }

void registerOpenXlaTritonPipelines() {
  PassPipelineRegistration<> mhlo(
      "openxla-nvgpu-triton-to-llvm-compilation-pipeline",
      "Runs the OpenXLA Triton to LLVM compilation pipeline",
      [](OpPassManager &passManager) {
        auto options = TritonOptions::FromFlags::get();
        buildTritonCompilationPipeline(passManager, options);
      });
}

}  // namespace openxla::compiler::nvgpu::triton
