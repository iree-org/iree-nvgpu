// Copyright 2023 The OpenXLA Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef OPENXLA_RUNTIME_NVGPU_CUDNN_MODULE_H_
#define OPENXLA_RUNTIME_NVGPU_CUDNN_MODULE_H_

#include "iree/base/api.h"
#include "iree/vm/api.h"

extern "C" iree_status_t openxla_nvgpu_cudnn_module_create(
    iree_vm_instance_t* instance, iree_allocator_t host_allocator,
    iree_vm_module_t** out_module);

extern "C" iree_status_t openxla_nvgpu_cudnn_module_register_types(
    iree_vm_instance_t* instance);

#endif  // OPENXLA_RUNTIME_NVGPU_CUDNN_MODULE_H_
