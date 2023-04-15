//===- CUDNNTypes.cpp - CUDNN dialect types -----------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "openxla/compiler/nvgpu/Dialect/CUDNN/IR/CUDNNTypes.h"

#include <cstdint>
#include <variant>

#include "llvm/ADT/TypeSwitch.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/Support/LogicalResult.h"
#include "openxla/compiler/nvgpu/Dialect/CUDNN/IR/CUDNNDialect.h"

using namespace mlir;
using namespace openxla::compiler::nvgpu::cudnn;

static ParseResult parseDimensionList(AsmParser &parser,
                                      SmallVector<int64_t> &dims, Type &type) {
  if (failed(parser.parseDimensionList(dims, /*allowDynamic=*/true,
                                       /*withTrailingX=*/true)) ||
      failed(parser.parseType(type)))
    return failure();
  return success();
}

static void printDimensionList(AsmPrinter &printer, ArrayRef<int64_t> dims,
                               Type type) {
  auto print = [&](int64_t dim) {
    ShapedType::isDynamic(dim) ? printer << "?" : printer << dim;
  };
  llvm::interleave(dims, printer.getStream(), print, "x");
  if (!dims.empty()) printer << 'x';
  printer << type;
}

#include "openxla/compiler/nvgpu/Dialect/CUDNN/IR/CUDNNEnums.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "openxla/compiler/nvgpu/Dialect/CUDNN/IR/CUDNNTypes.cpp.inc"

void CUDNNDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "openxla/compiler/nvgpu/Dialect/CUDNN/IR/CUDNNTypes.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// !cudnn.tensor
//===----------------------------------------------------------------------===//

namespace openxla::compiler::nvgpu::cudnn {

bool TensorType::isOpaque() {
  return getShape().empty() && !getElementType() && !getLayout() &&
         !getStrides();
}

LogicalResult TensorType::verify(function_ref<InFlightDiagnostic()> emitError,
                                 ArrayRef<long> shape, Type elementType,
                                 std::optional<Layout> layout,
                                 AffineMap strides) {
  if (elementType && (shape.size() < 3 || shape.size() > 8))
    return emitError() << "cuDNN supports tensors of rank 3 to 8";

  if (layout && strides)
    return emitError() << "layout can't be defined together with strides";

  if (strides && strides.getNumDims() != shape.size())
    return emitError() << "number of strides dimensions must match tensor rank";

  // TODO(ezhulenev): We can relax this constraint, but we must verify that we
  // can compute strides array from an affine map for passing them to cuDNN.
  if (strides && !strides.isPermutation())
    return emitError() << "strides must be a permutation";

  return success();
}

void TensorType::print(AsmPrinter &printer) const {
  // Opaque cuDNN tensor type (`!cudnn.tensor`)
  if (!getElementType()) return;

  printer << "<";
  printDimensionList(printer, getShape(), getElementType());
  if (getLayout().has_value()) {
    printer << ", " << stringifyLayout(*getLayout());
  }
  if (getStrides()) {
    printer << ", affine_map<" << getStrides() << ">";
  }
  printer << '>';
}

Type TensorType::parse(mlir::AsmParser &parser) {
  // Opaque cuDNN tensor type (`!cudnn.tensor`)
  if (failed(parser.parseOptionalLess())) {
    return TensorType::get(parser.getContext());
  }

  SmallVector<int64_t> shape;
  Type elementType;
  // Parse tensor shape and element type.
  if (failed(parseDimensionList(parser, shape, elementType))) {
    return Type();
  }

  // Parse optional tensor layout.
  std::variant<std::monostate, Layout, AffineMap> tensor_layout;

  if (succeeded(parser.parseOptionalComma())) {
    //  Try to parse affine map defining strides.
    AffineMapAttr strides;
    if (auto parsed = parser.parseOptionalAttribute(strides);
        parsed.has_value() && succeeded(*parsed)) {
      tensor_layout = strides.getAffineMap();
    }

    // If affine map was not parsed, it means we must parse a layout.
    if (!strides) {
      auto parsed = FieldParser<Layout>::parse(parser);
      if (failed(parsed)) return Type();
      tensor_layout = *parsed;
    }
  }

  // Parse closing `>`.
  if (failed(parser.parseGreater())) return Type();

  if (auto *layout = std::get_if<Layout>(&tensor_layout))
    return TensorType::get(shape, elementType, *layout);

  if (auto *strides = std::get_if<AffineMap>(&tensor_layout))
    return TensorType::get(shape, elementType, *strides);

  return TensorType::get(shape, elementType);
}

}  // namespace openxla::compiler::nvgpu::cudnn
