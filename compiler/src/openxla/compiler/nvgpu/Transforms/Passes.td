// Copyright 2023 The OpenXLA Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

include "mlir/Pass/PassBase.td"

#ifndef OPENXLA_NVGPU_TRANSFORMS_PASSES_TD_
#define OPENXLA_NVGPU_TRANSFORMS_PASSES_TD_

def ConvertHLOToCUDNNPass : Pass<"openxla-nvgpu-convert-hlo-to-cudnn",
    "mlir::ModuleOp"> {
  let summary = "Converts stable HLO ops to cudnn ops";
  let constructor = [{
    ::openxla::compiler::nvgpu::cudnn::createConvertHLOToCUDNNPass()
  }];
  let dependentDialects = [
    "mlir::iree_compiler::IREE::HAL::HALDialect",
    "::openxla::compiler::nvgpu::cudnn::CUDNNDialect",
  ];
}

def FlowTransformInterpreterPass
    : Pass<"openxla-nvgpu-flow-transform-interpreter"> {
  let summary = "Runs the Transform dialect interpreter at the Flow level";
  let constructor = [{
    ::openxla::compiler::nvgpu::createFlowTransformInterpreterPass()
  }];
  let options = [
    Option<"transformFileName", "transform-file-name", "std::string",
            /*default=*/"\"\"",
            "Optional filename containing a transform dialect specification to "
            "apply. If left empty, the IR is assumed to contain one top-level "
            "transform dialect operation somewhere in the module.">,
    Option<"transformLibraryFileName",
           "transform-library-file-name",
           "std::string",
           /*default=*/"\"\"",
           "If non-empty, the name of the file containing definitions of "
           "external symbols referenced in the transform script. "
           "These definitions will be used to replace declarations.">,
    Option<"debugPayloadRootTag", "debug-payload-root-tag", "std::string",
            /*default=*/"\"\"",
            "Select the operation with 'transform.target_tag' attribute having "
            "the given value as payload IR root. This allows user control on "
            "what operation to transform in debug mode, without requiring "
            "intimate knowledge of the IREE nested pass pipeline.\\n"
            "If empty (normal operation mode), select the pass anchor "
            "operation in the IREE pipeline, as the payload IR root.">,
    Option<"debugTransformRootTag", "debug-transform-root-tag", "std::string",
            /*default=*/"\"\"",
            "Select the operation with 'transform.target_tag' attribute having "
            "the given value as container IR for top-level transform ops. This "
            "allows user control on what transformation to apply in debug "
            "mode, without requiring intimate knowledge of the IREE nested "
            "pass pipeline.\\n"
            "If empty (normal operation mode), select the container of the "
            "top-level transform op.">,
    Option<"debugEmitRemarkOnMatch", "debug-emit-remark-on-match", "bool",
            "false",
            "Emit remark whenever a subgraph matches a pattern.">,
    Option<"debugPrintConstructedModule",
           "debug-print-constructed-module",
           "bool",
           "false",
           "Print the transform module constructed on-the-fly by the pass.">,
    Option<"enableReductionDispatchFormation",
           "enable-reduction-dispatch-formation",
           "bool",
           "false",
           "Enable TD-driven dispatch region formation for reductions.">,
    Option<"enableMatmulPadSplitK",
           "enable-matmul-pad-split-k",
           "bool",
           "false",
           "Enable TD-driven graph-level matmul padding and split-K.">
  ];
}

#endif // OPENXLA_NVGPU_TRANSFORMS_PASSES_TD_
