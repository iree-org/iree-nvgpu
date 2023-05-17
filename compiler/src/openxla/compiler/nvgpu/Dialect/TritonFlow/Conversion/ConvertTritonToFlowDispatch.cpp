// Copyright 2023 The OpenXLA Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "openxla/compiler/nvgpu/Dialect/TritonFlow/Conversion/ConvertTritonToFlowDispatch.h"

#include <cstdint>
#include <cstdio>
#include <functional>
#include <memory>
#include <optional>
#include <string>

#include "compiler/src/iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "compiler/src/iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"
#include "iree/compiler/Utils/OptionUtils.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/IndexToLLVM/IndexToLLVM.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Value.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"
#include "openxla/compiler/nvgpu/Dialect/TritonFlow/IR/TritonFlowOps.h"
#include "triton/Conversion/TritonGPUToLLVM/TritonGPUToLLVMPass.h"
#include "triton/Conversion/TritonToTritonGPU/TritonToTritonGPUPass.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Target/LLVMIR/LLVMIRTranslation.h"
#include "triton/Target/PTX/PTXTranslation.h"

IREE_DEFINE_COMPILER_OPTION_FLAGS(
    openxla::compiler::nvgpu::tritonflow::TritonOptions);

namespace openxla::compiler::nvgpu::tritonflow {

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::iree_compiler;

//===----------------------------------------------------------------------===//
// TritonOptions
//===----------------------------------------------------------------------===//

void TritonOptions::bindOptions(OptionsBinder &binder) {
  static llvm::cl::OptionCategory category("OpenXLA Triton compiler options");

  binder.opt<int32_t>(
      "openxla-triton-compute-capability", compute_capability,
      llvm::cl::desc("Compute capability for compiling Triton programs."),
      llvm::cl::cat(category));

  binder.opt<int32_t>(
      "openxla-triton-num-warps", num_warps,
      llvm::cl::desc("Number of warps for compiling Triton programs."),
      llvm::cl::cat(category));

  binder.opt<int32_t>(
      "openxla-triton-num-stages", num_stages,
      llvm::cl::desc("Number of stages for compiling Triton programs."),
      llvm::cl::cat(category));
}

//===----------------------------------------------------------------------===//
// Compilation pipeline from Triton IR to LLVM IR
//===----------------------------------------------------------------------===//

void buildTritonCompilationPipeline(mlir::OpPassManager &pm,
                                    const TritonOptions &opts) {
  // Based on optimize_ttir() in
  // @triton//:python/triton/compiler/compiler.py
  pm.addPass(createInlinerPass());
  pm.addPass(createCombineOpsPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
  pm.addPass(createLoopInvariantCodeMotionPass());
  pm.addPass(createSymbolDCEPass());

  // Based on ttir_to_ttgir() in
  // @triton//:python/triton/compiler/compiler.py
  pm.addPass(createConvertTritonToTritonGPUPass(opts.num_warps));

  // Based on optimize_ttgir() in
  // @triton//:python/triton/compiler/compiler.py
  pm.addPass(createTritonGPUCoalescePass());
  pm.addPass(createTritonGPURemoveLayoutConversionsPass());
  pm.addPass(createTritonGPUAccelerateMatmulPass(opts.compute_capability));
  pm.addPass(createTritonGPURemoveLayoutConversionsPass());
  pm.addPass(createTritonGPUOptimizeDotOperandsPass());
  pm.addPass(createTritonGPUPipelinePass(opts.num_stages));
  pm.addPass(createTritonGPUPrefetchPass());
  pm.addPass(createTritonGPUOptimizeDotOperandsPass());
  pm.addPass(createTritonGPURemoveLayoutConversionsPass());
  pm.addPass(createTritonGPUDecomposeConversionsPass());
  pm.addPass(createTritonGPUReorderInstructionsPass());
  pm.addPass(createCSEPass());
  pm.addPass(createSymbolDCEPass());

  // Based on translateTritonGPUToLLVMIR() in
  // @triton//:lib/Target/LLVMIR/LLVMIRTranslation.cpp
  pm.addPass(createConvertSCFToCFPass());
  pm.addPass(createConvertIndexToLLVMPass());
  pm.addPass(createConvertTritonGPUToLLVMPass(opts.compute_capability));
  pm.addPass(createArithToLLVMConversionPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
  pm.addPass(createSymbolDCEPass());
}

namespace {

//===----------------------------------------------------------------------===//
// Lowering from Triton to PTX
//===----------------------------------------------------------------------===//

// TODO(ezhulenev): We need a better way to run Triton compilation pipeline as a
// part of end-to-end IREE compilation flow. Current implementation relies on
// constructing a separate MLIR context for running Triton to LLVM IR lowering.
// This certainly can be done in a proper MLIR way by constructing nested pass
// pipelines matching nestied MLIR modules with Triton function. However we are
// in early proof of concept stage, so we are doing ugly workarounds here.
static FailureOr<std::string> compileTritonFunctionToPTX(
    FuncOp fn, const TritonOptions &opts) {
  // If we try to do lowering using default MLIR context, it all crashes because
  // we try to register new dialects in the middle of executing conversion pass.
  MLIRContext tritonCtx(MLIRContext::Threading::DISABLED);
  tritonCtx.appendDialectRegistry(fn->getContext()->getDialectRegistry());

  // Serialize function to string to "clone" it into a new context.
  std::string fn_serialized;
  {
    llvm::raw_string_ostream os(fn_serialized);
    fn->print(os, OpPrintingFlags());
  }

  // Parse Triton function back to MLIR operation.
  llvm::SourceMgr src;
  src.AddNewSourceBuffer(
      llvm::MemoryBuffer::getMemBuffer(
          fn_serialized, (fn.getSymName() + ".serialized.mlir").str()),
      llvm::SMLoc());
  auto tritonModule = parseSourceFile<ModuleOp>(src, &tritonCtx);

  // Create a pipeline for lowering from Triton to LLVM.
  PassManager pm(&tritonCtx, ModuleOp::getOperationName(),
                 OpPassManager::Nesting::Implicit);
  buildTritonCompilationPipeline(pm, opts);

  // Lower Triton function to LLVM IR.
  if (failed(pm.run(*tritonModule))) return failure();

  // Translate Triton LLVM IR module to LLVM proper.
  auto llvmContext = std::make_unique<llvm::LLVMContext>();
  std::unique_ptr<llvm::Module> llvmTritonModule =
      translateLLVMToLLVMIR(llvmContext.get(), *tritonModule,
                            /*isROCM=*/false);

  return ::triton::translateLLVMIRToPTX(
      *llvmTritonModule, opts.compute_capability, opts.ptx_version);
}

//===----------------------------------------------------------------------===//
// triton.dispatch
//===----------------------------------------------------------------------===//

using llvm::sys::fs::TempFile;

struct ConvertTritonFlowDispatchOp : public OpConversionPattern<DispatchOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      DispatchOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    MLIRContext *ctx = getContext();

    // TODO(ezhulenev): Should options be configured at a function level?
    auto opts = TritonOptions::FromFlags::get();

    // TODO(ezhulenev): This is a performance footgun, fix it!
    SymbolTable sym_table(op->getParentOfType<ModuleOp>());
    auto fn = sym_table.lookupNearestSymbolFrom<FuncOp>(op, op.getCalleeAttr());

    // Get a PTX module from the Triton callee.
    FailureOr<std::string> ptx = compileTritonFunctionToPTX(fn, opts);
    if (failed(ptx))
      return rewriter.notifyMatchFailure(
          op, "Failed to lower Triton function to LLVM IR");

    // TODO(ezhulenev): Executable object supports `data` attribute, check if
    // it works for embedding Triton PTX.

    // Write PTX to a temp file, so that we can constuct an HAL executable.
    SmallVector<char, 128> tmpPath;
    llvm::sys::path::system_temp_directory(/*erasedOnReboot=*/true, tmpPath);
    llvm::sys::path::append(tmpPath, fn.getSymName() + ".executable.%%%%%.ptx");
    llvm::Expected<TempFile> tmpFile = TempFile::create(tmpPath);
    if (!tmpFile)
      return rewriter.notifyMatchFailure(
          op, "Could not open temporary file to save the PTX blob");

    llvm::raw_fd_ostream(tmpFile->FD, /*shouldClose=*/false) << ptx;

    // Construct attributes describing Triton PTX executable.
    auto target = getExecutableTarget(ctx, opts);
    auto executableObject = getExecutableObject(ctx, tmpFile->TmpName);
    auto executableObjects = getExecutableObjects(target, executableObject);

    // Keep temp file because we'll have to read it back at later compilation
    // stage when lowering HAL to VM.
    if (tmpFile->keep())
      return rewriter.notifyMatchFailure(
          op, "Could not keep a temporary file with PTX blob");

    // Create `hal.executable.source` operation holding executable object (PTX).
    ImplicitLocOpBuilder b(fn->getLoc(), rewriter);
    b.setInsertionPointAfter(fn);

    auto executableSource = b.create<IREE::HAL::ExecutableSourceOp>(
        b.getStringAttr("private"),
        b.getStringAttr(fn.getSymName() + ".executable"), executableObjects);

    // TODO(ezhulenev): This us hardcoded for the block size used by the Triton
    // kernel. We have to infer this from the Triton kernel definition.
    auto workgroups = b.getIndexArrayAttr({64, 1, 1});

    // Create `hal.executable.export` operation to export Triton entrypoint.
    b.setInsertionPointToStart(&executableSource.getBody().emplaceBlock());
    auto executableExport = b.create<IREE::HAL::ExecutableExportOp>(
        fn.getSymNameAttr(),
        /*ordinal=*/b.getIndexAttr(0), getPipelineLayout(ctx), workgroups,
        /*subgroup_size=*/IntegerAttr(),
        /*workgroup_local_memory=*/IntegerAttr());
    b.create<IREE::HAL::ExecutableSourceEndOp>();

    // Replace `triton.dispatch` with a `flow.dispatch` operation.
    auto executableEntryPoint = SymbolRefAttr::get(
        ctx, executableSource.getSymName(),
        FlatSymbolRefAttr::get(ctx, executableExport.getSymName()));

    rewriter.replaceOpWithNewOp<IREE::Flow::DispatchOp>(
        op, executableEntryPoint, adaptor.getGrid(), op->getResultTypes(),
        /*resultDims=*/ValueRange(), adaptor.getArguments(),
        /*argumentDims=*/ValueRange(), /*tiedOperands=*/ArrayRef<int64_t>());

    // TODO(ezhulenev): It is not safe in general, but currently we assume that
    // we have 1-to-1 mapping between Triton dispatch and Triton function.
    rewriter.eraseOp(fn);

    return success();
  }

 private:
  // TODO(ezhulenev): We should support compiling Triton IR for multiple
  // architectures and dispatching them at run time based on the hardware.

  IREE::HAL::ExecutableTargetAttr getExecutableTarget(
      MLIRContext *ctx, const TritonOptions &opts) const {
    Builder b(ctx);

    SmallVector<NamedAttribute> config = {
        {b.getStringAttr("target_arch"),
         b.getStringAttr(llvm::formatv("sm_{0}", opts.compute_capability))}};

    return IREE::HAL::ExecutableTargetAttr::get(
        ctx, b.getStringAttr("cuda"), b.getStringAttr("cuda-nvptx-fb"),
        b.getDictionaryAttr(config));
  }

  IREE::HAL::ExecutableObjectAttr getExecutableObject(MLIRContext *ctx,
                                                      StringRef path) const {
    Builder b(ctx);
    return IREE::HAL::ExecutableObjectAttr::get(ctx, b.getStringAttr(path),
                                                /*data=*/nullptr);
  }

  IREE::HAL::DeviceTargetAttr getDeviceTarget(
      IREE::HAL::ExecutableTargetAttr target) const {
    Builder b(target.getContext());

    SmallVector<NamedAttribute> config = {
        {b.getStringAttr("legacy_sync"), b.getUnitAttr()},
        {b.getStringAttr("executable_targets"), b.getArrayAttr(target)}};

    return IREE::HAL::DeviceTargetAttr::get(
        b.getContext(), b.getStringAttr("cuda"), b.getDictionaryAttr(config));
  }

  IREE::HAL::ExecutableObjectsAttr getExecutableObjects(
      IREE::HAL::ExecutableTargetAttr target,
      IREE::HAL::ExecutableObjectAttr executable) const {
    Builder b(target.getContext());
    return IREE::HAL::ExecutableObjectsAttr::get(
        b.getContext(), b.getArrayAttr(target),
        b.getArrayAttr(b.getArrayAttr(executable)));
  }

  // TODO(ezhulenev): This pipeline layout is hardcoded for an `add_kernel`
  // triton kernel. We have to infer it from the original Triton function
  // signature.
  IREE::HAL::PipelineLayoutAttr getPipelineLayout(MLIRContext *ctx) const {
    Builder b(ctx);

    auto binding0 = IREE::HAL::DescriptorSetBindingAttr::get(
        ctx, /*ordinal=*/0, IREE::HAL::DescriptorType::StorageBuffer,
        IREE::HAL::DescriptorFlags::ReadOnly);

    auto binding1 = IREE::HAL::DescriptorSetBindingAttr::get(
        ctx, /*ordinal=*/1, IREE::HAL::DescriptorType::StorageBuffer,
        IREE::HAL::DescriptorFlags::ReadOnly);

    auto binding2 = IREE::HAL::DescriptorSetBindingAttr::get(
        ctx, /*ordinal=*/2, IREE::HAL::DescriptorType::StorageBuffer,
        std::nullopt);

    auto descriptorSetLayout = IREE::HAL::DescriptorSetLayoutAttr::get(
        ctx, /*ordinal=*/0, {binding0, binding1, binding2});

    return IREE::HAL::PipelineLayoutAttr::get(ctx, /*pushConstants=*/1,
                                              descriptorSetLayout);
  }
};

}  // namespace

void populateTritonToFlowDispatchPatterns(TypeConverter &typeConverter,
                                          RewritePatternSet &patterns) {
  MLIRContext *ctx = patterns.getContext();
  patterns.insert<ConvertTritonFlowDispatchOp>(typeConverter, ctx);
}

}  // namespace openxla::compiler::nvgpu::tritonflow
