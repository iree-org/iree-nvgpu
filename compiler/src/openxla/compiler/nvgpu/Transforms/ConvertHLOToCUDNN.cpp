// Copyright 2023 The OpenXLA Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <algorithm>
#include <iterator>
#include <numeric>

#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "openxla/compiler/nvgpu/Dialect/CUDNN/IR/CUDNNDialect.h"
#include "openxla/compiler/nvgpu/Dialect/CUDNN/IR/CUDNNOps.h"
#include "openxla/compiler/nvgpu/Dialect/CUDNN/IR/CUDNNTypes.h"
#include "openxla/compiler/nvgpu/Transforms/Passes.h"
#include "stablehlo/dialect/StablehloOps.h"

#define GEN_PASS_DEF_CONVERTHLOTOCUDNNPASS
#include "openxla/compiler/nvgpu/Transforms/Passes.h.inc"

using namespace mlir;
using namespace mlir::iree_compiler;

namespace openxla::compiler::nvgpu::cudnn {

namespace {
class CudnnHandleCache {
 public:
  IREE::Util::GlobalOp getGlobalHandle(PatternRewriter &rewriter, Location loc,
                                       ModuleOp m, StringRef baseName) {
    if (globalHandle) return *globalHandle;

    // Create global handle before any other initialization.
    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPointToStart(m.getBody());
    static constexpr char globalHandleName[] = "cudnn.shared.handle";
    auto handleTy = rewriter.getType<HandleType>();
    globalHandle = rewriter.create<IREE::Util::GlobalOp>(
        loc, globalHandleName, /*isMutable=*/false, handleTy);

    // Create global handle initializer.
    auto initializer = rewriter.create<IREE::Util::InitializerOp>(loc);
    rewriter.setInsertionPointToStart(initializer.addEntryBlock());
    auto device = rewriter.create<IREE::HAL::ExSharedDeviceOp>(loc);
    auto handle = rewriter.create<HandleOp>(loc, device);
    rewriter.create<IREE::Util::GlobalStoreOp>(loc, handle, *globalHandle);
    rewriter.create<IREE::Util::InitializerReturnOp>(loc);

    return *globalHandle;
  }

 private:
  std::optional<IREE::Util::GlobalOp> globalHandle;
};
}  // namespace

// Helper functions for permutations.

static SmallVector<int64_t> getTensorLayoutPermutation(
    ArrayRef<int64_t> srcSpatialDims, int64_t srcBatchDim,
    int64_t srcFeatureDim, Layout dst) {
  assert(srcSpatialDims.size() == 2 && "expect 2 spatial dims");
  switch (dst) {
    case Layout::NHWC:
      return {srcBatchDim, srcSpatialDims[0], srcSpatialDims[1], srcFeatureDim};
    case Layout::NCHW:
      return {srcBatchDim, srcFeatureDim, srcSpatialDims[0], srcSpatialDims[1]};
    default:
      llvm_unreachable("not a kernel layout");
  }
}

static SmallVector<int64_t> getKernelLayoutPermutation(
    ArrayRef<int64_t> srcSpatialDims, int64_t srcOutputFeatureDim,
    int64_t srcInputFeatureDim, Layout dst) {
  assert(srcSpatialDims.size() == 2 && "expect 2 spatial dims");
  switch (dst) {
    case Layout::KHWC:
      return {{srcOutputFeatureDim, srcSpatialDims[0], srcSpatialDims[1],
               srcInputFeatureDim}};
    case Layout::KCHW:
      return {{srcOutputFeatureDim, srcInputFeatureDim, srcSpatialDims[0],
               srcSpatialDims[1]}};
    default:
      llvm_unreachable("not a kernel layout");
  }
}

static SmallVector<int64_t> getReversePermutation(
    ArrayRef<int64_t> permutation) {
  int64_t n = permutation.size();
  SmallVector<int64_t> reverse(n);
  for (int64_t i = 0; i < n; ++i) reverse[permutation[i]] = i;
  return reverse;
}

static bool isIdentityPermutation(const ArrayRef<int64_t> permutation) {
  int64_t n = permutation.size();
  for (int64_t i = 0; i < n; ++i) {
    if (permutation[i] != i) return false;
  }
  return true;
}

static SmallVector<int64_t> applyPermutation(SmallVector<int64_t> &permutation,
                                             ArrayRef<int64_t> argument) {
  int64_t n = argument.size();
  SmallVector<int64_t> result(n);
  for (int64_t i = 0; i < n; ++i) result[i] = argument[permutation[i]];
  return result;
}

static RankedTensorType getPermutedTensorType(
    Type originalTy, SmallVector<int64_t> &permutation) {
  auto rankedTy = originalTy.cast<RankedTensorType>();
  SmallVector<int64_t> permutedShape =
      applyPermutation(permutation, rankedTy.getShape());
  return RankedTensorType::get(permutedShape, rankedTy.getElementType());
}

static CudnnTensorType getCudnnTensorType(Type originalTy, Layout layout) {
  auto rankedTy = originalTy.cast<RankedTensorType>();
  ArrayRef<int64_t> shape = rankedTy.getShape();

  // Infer shape based on layout.
  SmallVector<int64_t> cudnnTensorShape;
  switch (layout) {
    case Layout::NCHW:
    case Layout::KCHW:
      cudnnTensorShape = {shape[0], shape[1], shape[2], shape[3]};
      break;
    case Layout::NHWC:
    case Layout::KHWC:
      cudnnTensorShape = {shape[0], shape[3], shape[1], shape[2]};
      break;
  }

  Type elemTy = rankedTy.getElementType();
  return CudnnTensorType::get(cudnnTensorShape, elemTy, layout);
}

// Permuted value to represent CUDNN graph operands, which may need permutation
// to match the inner layout.
namespace {
struct PermutedValue {
  Value value;
  SmallVector<int64_t> permutation;
};
}  // namespace

// Declare for recursive use.
static Value getValueInCudnnGraphRecursively(
    Value originalValue, SmallVector<int64_t> &permutation, Layout layout,
    SmallVector<PermutedValue> &cuddnGraphOperands, Block *cudnnGraphBody,
    SmallVector<Operation *> &toBeErased, PatternRewriter &rewriter);

// Relu.

static FailureOr<double> matchRelu(stablehlo::ClampOp op,
                                   PatternRewriter &rewriter) {
  llvm::APFloat min = llvm::APFloat::IEEEdouble();
  if (!matchPattern(op.getMin().getDefiningOp(), m_ConstantFloat(&min))) {
    return rewriter.notifyMatchFailure(op, "expected constant min");
  }
  llvm::APFloat max = llvm::APFloat::IEEEdouble();
  if (!matchPattern(op.getMax().getDefiningOp(), m_ConstantFloat(&max)) ||
      !max.isNaN()) {
    return rewriter.notifyMatchFailure(op, "expected NaN max");
  }
  return min.convertToDouble();
}

static FailureOr<Value> tryGettingValueInCudnnGraphRecursivelyByFusingRelu(
    Value originalValue, SmallVector<int64_t> permutation, Layout layout,
    SmallVector<PermutedValue> &cuddnGraphOperands, Block *cudnnGraphBody,
    SmallVector<Operation *> &toBeErased, PatternRewriter &rewriter) {
  // Match relu.
  Operation *def = originalValue.getDefiningOp();
  auto clampOp = llvm::dyn_cast_or_null<stablehlo::ClampOp>(def);
  if (!clampOp || !clampOp->hasOneUse()) return failure();
  FailureOr<double> min = matchRelu(clampOp, rewriter);
  if (failed(min)) return failure();

  // Mark original op for ereasure.
  toBeErased.push_back(clampOp);

  // Recur for operands.
  Value operand = getValueInCudnnGraphRecursively(
      clampOp.getOperand(), permutation, layout, cuddnGraphOperands,
      cudnnGraphBody, toBeErased, rewriter);

  // Create relu op in CUDNN graph.
  Location loc = clampOp.getLoc();
  TensorType originalResultTy = clampOp.getType();
  Type resultTy = getCudnnTensorType(
      getPermutedTensorType(originalResultTy, permutation), layout);
  Type computeTy = originalResultTy.getElementType();
  return rewriter
      .create<ReluOp>(loc, resultTy, operand, computeTy, APFloat(*min))
      .getResult();
}

// Convolution.

static LogicalResult matchConvolution(stablehlo::ConvolutionOp op,
                                      PatternRewriter &rewriter) {
  // Check group counts.
  if (op.getBatchGroupCount() != 1) {
    return rewriter.notifyMatchFailure(op,
                                       "expected batch_group_count to be 1");
  }
  if (op.getFeatureGroupCount() != 1) {
    return rewriter.notifyMatchFailure(op,
                                       "expected feature_group_count to be 1");
  }
  if (op.getPrecisionConfig() &&
      llvm::any_of(*op.getPrecisionConfig(), [](Attribute a) {
        return a.cast<stablehlo::PrecisionAttr>().getValue() !=
               stablehlo::Precision::DEFAULT;
      })) {
    return rewriter.notifyMatchFailure(
        op, "expected no or default precision config");
  }

  // Check unit dilation.
  auto isNoneOrValue = [](std::optional<DenseIntElementsAttr> attr,
                          int64_t value) {
    return !attr || llvm::all_of(attr->getValues<APInt>(), [&](APInt value) {
      return value.getSExtValue() == value;
    });
  };
  if (!isNoneOrValue(op.getLhsDilation(), 1))
    return rewriter.notifyMatchFailure(op, "expected lhs dilation to be 1");
  if (!isNoneOrValue(op.getRhsDilation(), 1))
    return rewriter.notifyMatchFailure(op, "expected rhs dilation to be 1");

  // Check no window reversal.
  if (op.getWindowReversal() &&
      llvm::any_of(op.getWindowReversal()->getValues<bool>(),
                   [](bool reversal) { return reversal; })) {
    return rewriter.notifyMatchFailure(op, "expected no window reversal");
  }

  // Check 2D convolution.
  auto dims = op.getDimensionNumbers();
  if (dims.getInputSpatialDimensions().size() != 2 ||
      dims.getOutputSpatialDimensions().size() != 2 ||
      dims.getKernelSpatialDimensions().size() != 2) {
    return rewriter.notifyMatchFailure(op, "expected 2D convolution");
  }

  return success();
}

static FailureOr<Value>
tryGettingValueInCudnnGraphRecursivelyByFusingConvolution(
    Value originalValue, SmallVector<int64_t> permutation, Layout layout,
    SmallVector<PermutedValue> &cuddnGraphOperands, Block *cudnnGraphBody,
    SmallVector<Operation *> &toBeErased, PatternRewriter &rewriter) {
  // Match convolution.
  Operation *def = originalValue.getDefiningOp();
  auto convOp = llvm::dyn_cast_or_null<stablehlo::ConvolutionOp>(def);
  if (!convOp || !convOp->hasOneUse() ||
      failed(matchConvolution(convOp, rewriter)))
    return failure();

  // Mark original op for ereasure.
  toBeErased.push_back(convOp);

  // Recur for input operand.
  // Result layout is assumed to be the same as that of the operand (inferred in
  // runtime).
  Layout inputLayout = layout;
  auto dims = convOp.getDimensionNumbers();
  auto inputPermutation = getTensorLayoutPermutation(
      /*srcSpatialDims=*/dims.getInputSpatialDimensions(),
      /*srcBatchDim=*/dims.getInputBatchDimension(),
      /*srcFeatureDim=*/dims.getInputFeatureDimension(),
      /*dst=*/inputLayout);
  Value input = getValueInCudnnGraphRecursively(
      convOp.getLhs(), inputPermutation, inputLayout, cuddnGraphOperands,
      cudnnGraphBody, toBeErased, rewriter);

  // Recur for kernel operand.
  Layout kernelLayout = Layout::KHWC;
  auto kernelPermutation = getKernelLayoutPermutation(
      /*srcSpatialDims=*/dims.getKernelSpatialDimensions(),
      /*srcOutputFeatureDim=*/dims.getKernelOutputFeatureDimension(),
      /*srcInputFeatureDim=*/dims.getKernelInputFeatureDimension(),
      /*dst*/ kernelLayout);
  Value kernel = getValueInCudnnGraphRecursively(
      convOp.getRhs(), kernelPermutation, kernelLayout, cuddnGraphOperands,
      cudnnGraphBody, toBeErased, rewriter);

  // Create conv op in CUDNN graph.
  Location loc = convOp.getLoc();
  Type resultTy = getCudnnTensorType(
      getPermutedTensorType(convOp.getType(), permutation), layout);
  APFloat alpha(1.0f);
  APFloat beta(0.0f);
  int64_t numSpatialDims = dims.getInputSpatialDimensions().size();
  assert(numSpatialDims == 2 && "expect 2 spatial dims");
  auto getAttrOr = [](std::optional<DenseIntElementsAttr> attr, int64_t size,
                      int64_t value) -> SmallVector<int64_t> {
    if (!attr) return SmallVector<int64_t>(size, value);
    return llvm::to_vector(llvm::map_range(
        attr->getValues<APInt>(), [](APInt it) { return it.getSExtValue(); }));
  };
  SmallVector<int64_t> spatialStride =
      getAttrOr(convOp.getWindowStrides(), numSpatialDims, 1);
  SmallVector<int64_t> dilation =
      getAttrOr(convOp.getRhsDilation(), numSpatialDims, 1);
  SmallVector<int64_t> padding =
      getAttrOr(convOp.getPadding(), 2 * numSpatialDims, 0);
  assert(padding.size() == 2 * numSpatialDims);
  SmallVector<int64_t> prePadding, postPadding;
  prePadding.reserve(numSpatialDims);
  postPadding.reserve(numSpatialDims);
  for (int64_t i = 0; i < 2 * numSpatialDims;) {
    prePadding.push_back(padding[i++]);
    postPadding.push_back(padding[i++]);
  }
  return rewriter
      .create<ConvolutionOp>(loc, resultTy, input, kernel, alpha, beta,
                             numSpatialDims, spatialStride, prePadding,
                             postPadding, dilation)
      .getResult();
}

static Value getValueInCudnnGraphAsArg(
    Value originalValue, SmallVector<int64_t> &permutation, Layout layout,
    SmallVector<PermutedValue> &cuddnGraphOperands, Block *cudnnGraphBody) {
  Location loc = originalValue.getLoc();

  // Add the original value with its expected permutation to the CUDNN graph
  // operands. The value will be permuted to match the CUDNN layout and passed
  // in as an argument.
  PermutedValue permutedOperand = {/*value=*/originalValue,
                                   /*permutation=*/permutation};
  cuddnGraphOperands.push_back(permutedOperand);

  // Derive the inner CUDNN tensor type and add it as an argument.
  auto cudnnTy = getCudnnTensorType(
      getPermutedTensorType(originalValue.getType(), permutation), layout);
  return cudnnGraphBody->addArgument(cudnnTy, loc);
}

static Value getValueInCudnnGraphRecursively(
    Value originalValue, SmallVector<int64_t> &permutation, Layout layout,
    SmallVector<PermutedValue> &cuddnGraphOperands, Block *cudnnGraphBody,
    SmallVector<Operation *> &toBeErased, PatternRewriter &rewriter) {
  // Try fusing convolutions into the CUDNN graph.
  FailureOr<Value> fusedConvolutionResult =
      tryGettingValueInCudnnGraphRecursivelyByFusingConvolution(
          originalValue, permutation, layout, cuddnGraphOperands,
          cudnnGraphBody, toBeErased, rewriter);
  if (succeeded(fusedConvolutionResult)) return *fusedConvolutionResult;

  // Try fusing relus into the CUDNN graph.
  FailureOr<Value> fusedReluResult =
      tryGettingValueInCudnnGraphRecursivelyByFusingRelu(
          originalValue, permutation, layout, cuddnGraphOperands,
          cudnnGraphBody, toBeErased, rewriter);
  if (succeeded(fusedReluResult)) return *fusedReluResult;

  // Give up and get value as an operand.
  return getValueInCudnnGraphAsArg(originalValue, permutation, layout,
                                   cuddnGraphOperands, cudnnGraphBody);
}

Value findRootForCudnnGraphOutlining(Value candidate,
                                     PatternRewriter &rewriter) {
  // Follow supported cwise ops as long as they have unique ueses.
  if (!candidate.hasOneUse()) return candidate;
  Operation *user = *candidate.getUsers().begin();

  // Match and follow relu ops.
  if (auto clampOp = llvm::dyn_cast_or_null<stablehlo::ClampOp>(user)) {
    if (succeeded(matchRelu(clampOp, rewriter))) {
      return findRootForCudnnGraphOutlining(clampOp.getResult(), rewriter);
    }
  }

  // Give up and use inital candidate.
  return candidate;
}

GraphOp createCudnnGraphRecursively(
    StringRef baseName, Value originalValue, SmallVector<int64_t> &permutation,
    Layout layout, SmallVector<PermutedValue> &cuddnGraphOperands,
    SmallVector<Operation *> &toBeErased, PatternRewriter &rewriter) {
  // Create CUDNN graph op with empty region. Use a placeholder function type
  // and update it later when we know all the operands.
  auto funcOp =
      originalValue.getParentRegion()->getParentOfType<func::FuncOp>();
  SymbolTable symbolTable(funcOp->getParentOp());
  rewriter.setInsertionPoint(funcOp);
  Location loc = originalValue.getLoc();
  FunctionType placeholderFuncTy = rewriter.getFunctionType({}, {});
  auto graphOp = rewriter.create<GraphOp>(loc, placeholderFuncTy,
                                          /*arg_attrs=*/ArrayAttr{},
                                          /*res_attrs=*/ArrayAttr{});
  Block *cudnnGraphBody = graphOp.addEntryBlock();

  // Insert into symbol table.
  graphOp.setName(baseName);
  symbolTable.insert(graphOp);

  // Create CUDNN graph recursively and fuse producers greedily into it. Keep
  // track of the operands' permutations to guarantee the required CUDNN
  // layouts.
  rewriter.setInsertionPointToStart(cudnnGraphBody);
  Value result = getValueInCudnnGraphRecursively(
      originalValue, permutation, layout, cuddnGraphOperands, cudnnGraphBody,
      toBeErased, rewriter);
  rewriter.create<ReturnOp>(loc, result);

  // Update the graph's function type when all operand types are known.
  auto argumentTypes = cudnnGraphBody->getArgumentTypes();
  Type resultTy = result.getType();
  FunctionType funcTy = rewriter.getFunctionType(argumentTypes, {resultTy});
  graphOp.setFunctionType(funcTy);

  return graphOp;
}

namespace {
class OutlineCudnnGraphPattern
    : public OpRewritePattern<stablehlo::ConvolutionOp> {
 public:
  OutlineCudnnGraphPattern(MLIRContext *ctx,
                           std::shared_ptr<CudnnHandleCache> cudnnHandleCache,
                           PatternBenefit benefit = 1)
      : OpRewritePattern<stablehlo::ConvolutionOp>(ctx, benefit),
        cudnnHandleCache(cudnnHandleCache) {}
  LogicalResult matchAndRewrite(stablehlo::ConvolutionOp op,
                                PatternRewriter &rewriter) const override {
    // Match convolution to ensure that we include at least one op in the CUDNN
    // graph.
    if (failed(matchConvolution(op, rewriter))) return failure();

    // Choose a favorable CUDNN layout for the convolution.
    // Derive necessary transpose permutation to call into CUDNN.
    Layout layout = Layout::NHWC;
    auto dims = op.getDimensionNumbers();
    SmallVector<int64_t> permutation = getTensorLayoutPermutation(
        /*srcSpatialDims=*/dims.getOutputSpatialDimensions(),
        /*srcBatchDim=*/dims.getOutputBatchDimension(),
        /*srcFeatureDim=*/dims.getOutputFeatureDimension(), layout);
    SmallVector<int64_t> reverse = getReversePermutation(permutation);

    // Find the root for this CUDNN graph by following supported cwise ops as
    // long as they have unique ueses.
    Value root = findRootForCudnnGraphOutlining(op.getResult(), rewriter);

    // Create CUDNN graph recursively and fuse producers greedily into it. Keep
    // track of
    //  (i) the operands' permutations to guarantee the required CUDNN layouts,
    //  and
    // (ii) the original operations to be erased.
    StringRef baseName = op->getName().getStringRef();
    SmallVector<PermutedValue> cuddnGraphOperands;
    SmallVector<Operation *> toBeErased;
    GraphOp graphOp =
        createCudnnGraphRecursively(baseName, root, permutation, layout,
                                    cuddnGraphOperands, toBeErased, rewriter);

    // Get or create CUDNN handle.
    rewriter.setInsertionPointAfter(op);
    Location loc = op.getLoc();
    auto moduleOp = op->getParentOfType<ModuleOp>();
    IREE::Util::GlobalOp globalHandle =
        cudnnHandleCache->getGlobalHandle(rewriter, loc, moduleOp, baseName);
    auto handleTy = rewriter.getType<HandleType>();
    auto loadedHandle = rewriter.create<IREE::Util::GlobalLoadOp>(
        loc, handleTy, globalHandle.getSymNameAttr());

    // Transpose operands to the CUDNN call where needed.
    SmallVector<Value> transposedCuddnGraphOperands;
    for (const auto &it : cuddnGraphOperands) {
      Value transposedOperand = it.value;
      if (!isIdentityPermutation(it.permutation)) {
        transposedOperand = rewriter.create<stablehlo::TransposeOp>(
            loc, transposedOperand, rewriter.getI64TensorAttr(it.permutation));
      }
      transposedCuddnGraphOperands.push_back(transposedOperand);
    }

    // Materialize CUDNN call.
    Type resultTy = getPermutedTensorType(root.getType(), permutation);
    Value result =
        rewriter
            .create<CallOp>(loc, resultTy, graphOp.getName(), loadedHandle,
                            transposedCuddnGraphOperands)
            .getResults()
            .front();

    // Transpose result if needed.
    if (!isIdentityPermutation(reverse)) {
      result = rewriter.create<stablehlo::TransposeOp>(
          loc, result, rewriter.getI64TensorAttr(reverse));
    }
    assert(result.getType() == root.getType() && "expect same type as root");

    // Replace all uses and erase redundant ops.
    rewriter.replaceAllUsesWith(root, result);
    assert(toBeErased.size() > 0 && "expect at least one op in the graph");
    for (Operation *it : toBeErased) rewriter.eraseOp(it);

    return success();
  }

 private:
  std::shared_ptr<CudnnHandleCache> cudnnHandleCache;
};
}  // namespace

namespace {
class ConvertHLOToCUDNNPass
    : public ::impl::ConvertHLOToCUDNNPassBase<ConvertHLOToCUDNNPass> {
 public:
  void runOnOperation() override {
    // Populate patterns.
    MLIRContext *ctx = &getContext();
    ModuleOp m = getOperation();
    auto cudnnHandleCache = std::make_shared<CudnnHandleCache>();
    RewritePatternSet patterns(ctx);
    patterns.add<OutlineCudnnGraphPattern>(ctx, cudnnHandleCache);

    // Apply patterns.
    if (failed(applyPatternsAndFoldGreedily(m, std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};
}  // namespace

std::unique_ptr<OperationPass<ModuleOp>> createConvertHLOToCUDNNPass() {
  return std::make_unique<ConvertHLOToCUDNNPass>();
}

}  // namespace openxla::compiler::nvgpu::cudnn
