#!/bin/bash

# Copyright 2019 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Build and run tests that do not rely on a device being present.

set -xeuo pipefail

BUILD_DIR="${1:-${IREE_BUILD_DIR:-build}}"
CMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE:-Release}"
ENABLE_ASSERTIONS="${ENABLE_ASSERTIONS:-ON}"
CMAKE_BIN="${CMAKE_BIN:-$(which cmake)}"
"${CMAKE_BIN}" --version
ninja --version
python3 --version
echo "Current directory: ${PWD}"
ls -l

declare -a CMAKE_ARGS=(
  "-G" "Ninja"
  "-B" "${BUILD_DIR}"
  "-S" "$PWD/openxla-nvgpu"

  "-DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}"
  "-DIREE_ENABLE_ASSERTIONS=${ENABLE_ASSERTIONS}"

  # Use `lld` for faster linking.
  "-DIREE_ENABLE_LLD=ON"

  # Disable device tests.
  "-DOPENXLA_NVGPU_INCLUDE_DEVICE_TESTS=OFF"

  # Disable CUDNN runtime component for now.
  # TODO: Figure out how to get CUDNN in the docker containers and enable.
  "-DOPENXLA_NVGPU_BUILD_RUNTIME_CUDNN=OFF"
)

"$CMAKE_BIN" "${CMAKE_ARGS[@]}"

# We first run all tests, which implicitly builds needed deps.
# That way, we know that the test deps are declared properly.
echo "Building openxla-nvgpu-run-tests"
echo "--------------------------------"
"$CMAKE_BIN" --build "${BUILD_DIR}" --target openxla-nvgpu-run-tests -- -k 0

# Then followup with building all.
echo "Building all"
echo "------------"
"$CMAKE_BIN" --build "${BUILD_DIR}" -- -k 0
