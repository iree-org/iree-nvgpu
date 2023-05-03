# Copyright 2023 The OpenXLA Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

cmake_minimum_required(VERSION 3.21...3.24)

project(OPENXLA_NVGPU)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(OPENXLA_NVGPU_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR})

# TODO: Fix this once the project is slotted into place.
if(NOT IREE_ROOT_DIR)
  set(IREE_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../iree")
endif()

set(IREE_CXX_STANDARD ${CMAKE_CXX_STANDARD})

# Enable testing before any subdirectories are added.
enable_testing()

#-------------------------------------------------------------------------------
# Customize IREE defaults.
#-------------------------------------------------------------------------------

option(IREE_BUILD_COMPILER "Disable compiler for runtime-library build" ON)
option(IREE_BUILD_SAMPLES "Disable samples for runtime-library build" OFF)
option(IREE_HAL_DRIVER_DEFAULTS "Disable all HAL drivers by default" OFF)
option(IREE_HAL_DRIVER_CUDA "Enables the 'cuda' runtime HAL driver" ON)
option(IREE_TARGET_BACKEND_DEFAULTS "Disables target backend" OFF)
option(IREE_TARGET_BACKEND_CUDA "Enables CUDA target backend" ON)
option(IREE_COMPILER_BUILD_SHARED_LIBS "Enables shared libraries in the compiler by default" ON)

#-------------------------------------------------------------------------------
# Add OpenXLA Nvgpu compiler plugin to IREE compiler tools.
#-------------------------------------------------------------------------------

set(IREE_COMPILER_PLUGIN_PATHS "${CMAKE_CURRENT_SOURCE_DIR}" CACHE STRING "OpenXLA Nvgpu plugins")

#-------------------------------------------------------------------------------
# Add OpenXLA Nvgpu runtime modules too IREE runtime tools.
#-------------------------------------------------------------------------------

set(IREE_EXTERNAL_TOOLING_MODULES "cudnn" CACHE STRING "OpenXLA Nvgpu cuDNN module")

set(IREE_EXTERNAL_TOOLING_MODULE_CUDNN_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/runtime/src/openxla/runtime/nvgpu" CACHE STRING "OpenXLA Nvgpu cuDNN module source dir")
set(IREE_EXTERNAL_TOOLING_MODULE_CUDNN_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/runtime/src/openxla/runtime/nvgpu" CACHE STRING "OpenXLA Nvgpu cuDNN module binary dir")
set(IREE_EXTERNAL_TOOLING_MODULE_CUDNN_TARGET "openxla_runtime_nvgpu_cudnn_module" CACHE STRING "OpenXLA Nvgpu cuDNN module target")
set(IREE_EXTERNAL_TOOLING_MODULE_CUDNN_NAME "cudnn" CACHE STRING "OpenXLA Nvgpu cuDNN module name")
set(IREE_EXTERNAL_TOOLING_MODULE_CUDNN_REGISTER_TYPES "openxla_nvgpu_cudnn_module_register_types" CACHE STRING "OpenXLA Nvgpu cuDNN module type registration")
set(IREE_EXTERNAL_TOOLING_MODULE_CUDNN_CREATE "openxla_nvgpu_cudnn_module_create" CACHE STRING "OpenXLA Nvgpu cuDNN module creator")

#-------------------------------------------------------------------------------
# Set up IREE toolchain with OpenXLA Nvgpu modules.
#-------------------------------------------------------------------------------

add_subdirectory("${IREE_ROOT_DIR}" "iree_core")

# Handle various global definitions that need to be set at the global
# toolchain level.
iree_setup_toolchain()

#-------------------------------------------------------------------------------
# OpenXLA NVGPU Runtime.
#
# Integration of NVIDIA libraries with IREE runtime via custom VM modules.
#-------------------------------------------------------------------------------

add_subdirectory(runtime)
add_subdirectory(tools)

#-------------------------------------------------------------------------------
# NVIDIA dependencies
#-------------------------------------------------------------------------------

add_subdirectory(build_tools/third_party/cudnn-frontend EXCLUDE_FROM_ALL)
