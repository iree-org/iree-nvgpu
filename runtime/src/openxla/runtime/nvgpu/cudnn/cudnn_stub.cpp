// Copyright 2023 The OpenXLA Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "openxla/runtime/nvgpu/cudnn/cudnn_stub.h"

#include "openxla/runtime/nvgpu/cudnn/dynamic_symbols.h"

namespace openxla::runtime::nvgpu {

static thread_local openxla_cudnn_dynamic_symbols_t* cudnn_syms = nullptr;

ScopedCudnnStubs::ScopedCudnnStubs(openxla_cudnn_dynamic_symbols_t* syms)
    : syms_(cudnn_syms) {
  cudnn_syms = syms;
}

ScopedCudnnStubs::~ScopedCudnnStubs() { cudnn_syms = syms_; }

openxla_cudnn_dynamic_symbols_t* ScopedCudnnStubs::syms() { return cudnn_syms; }

}  // namespace openxla::runtime::nvgpu
