// Copyright 2023 The OpenXLA Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

include "mlir/Pass/PassBase.td"

#ifndef OPENXLA_NVGPU_TRANSFORMS_PASSES_TD_
#define OPENXLA_NVGPU_TRANSFORMS_PASSES_TD_

def ConvertMHLOToCUDNN : Pass<"openxla-nvgpu-convert-mhlo-to-cudnn", "mlir::ModuleOp"> {
  let summary = "Converts mhlo ops to cudnn ops";
  let constructor = [{
    ::openxla::compiler::nvgpu::cudnn::createConvertMHLOToCUDNNPass()
  }];
  let dependentDialects = [
    "::openxla::compiler::nvgpu::cudnn::CUDNNDialect",
  ];
}

#endif // OPENXLA_NVGPU_TRANSFORMS_PASSES_TD_
