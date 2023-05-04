# OpenXLA NVIDIA GPU Compiler and Runtime

This project contains the compiler and runtime plugins enabling specialized
targeting of the OpenXLA platform to NVIDIA GPUs. It builds on top of the
core IREE toolkit.

## Development setup

The project can be built either as part of IREE by manually specifying
plugin paths via `-DIREE_COMPILER_PLUGIN_PATHS`, or for development tailored
to NVIDIA GPUs specifically, can be built directly:

```
cmake -GNinja -B build/ -S . \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DIREE_ENABLE_ASSERTIONS=ON \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DIREE_ENABLE_LLD=ON

# Recommended:
# -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache
```

Note that you will need a check-out of the IREE codebase in `../iree` relative
to the directory where the `openxla-nvgpu` compiler was checked out. Refer to
the IREE [getting
started](https://openxla.github.io/iree/building-from-source/getting-started/)
guide for details of how to set this up.

## Installing dependencies

You must have a CUDA Toolkit installed together with a cuDNN ([see
instructions](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html#installlinux-tar)).

On Linux platform path to `libcudnn.so` should be added to `LD_LIBRARY_PATH`.

```
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/cuda/lib64
```

## Running cuDNN runtime tests

Some of the tests can run only on an Ampere+ devices because they rely on the
[cuDNN runtime fusion engine](https://docs.nvidia.com/deeplearning/cudnn/developer-guide/index.html#runtime-fusion-engine).

```
cmake --build build
ctest --test-dir build -R openxla/runtime/nvgpu/test/
```
