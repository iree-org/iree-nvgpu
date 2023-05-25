iree_add_all_subdirs()

iree_cc_library(
  NAME
    IR
  HDRS
    "TritonFlowDialect.h"
    "TritonFlowOps.h"
  TEXTUAL_HDRS
    "TritonFlowDialect.cpp.inc"
    "TritonFlowDialect.h.inc"
    "TritonFlowOps.cpp.inc"
    "TritonFlowOps.h.inc"
  SRCS
    "TritonFlowDialect.cpp"
    "TritonFlowOps.cpp"
  DEPS
    ::TritonFlowOpsGen
    LLVMSupport
    MLIRIR
    MLIRSupport
    iree::compiler::Dialect::Util::IR
    openxla::compiler::nvgpu::defs
    # TODO(ezhulenev): Triton dependencies are not part of bazel build.
    triton::TritonIR
  PUBLIC
)

iree_tablegen_library(
  NAME
    TritonFlowOpsGen
  TD_FILE
    "TritonFlowOps.td"
  OUTS
    --gen-dialect-decls --dialect=triton TritonFlowDialect.h.inc
    --gen-dialect-defs --dialect=triton TritonFlowDialect.cpp.inc
    --gen-op-decls TritonFlowOps.h.inc
    --gen-op-defs TritonFlowOps.cpp.inc
)
