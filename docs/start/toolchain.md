# Build the toolchain

The supported setup uses SuperNPUBench inside the LinxISA superproject. Build
the in-tree LLVM fork, musl sysroot, and QEMU so their revisions stay aligned.

## Prerequisites

Install Git, CMake, Ninja, Python 3, a host C/C++ compiler, and the build
dependencies required by LLVM and QEMU. Then clone every submodule:

```bash
git clone --recurse-submodules https://github.com/LinxISA/linx-isa.git
cd linx-isa
git submodule sync --recursive
git submodule update --init --recursive
export LINXISA_ROOT="$PWD"
export SUPERNPU_ROOT="$LINXISA_ROOT/workloads/SuperNPUBench"
```

All remaining commands on this page run from `$LINXISA_ROOT`.

## Build Linx LLVM

Configure only the projects and target needed by the benchmark lane:

```bash
cmake -S compiler/llvm/llvm \
  -B compiler/llvm/build-linxisa-clang \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_PROJECTS="clang;lld" \
  -DLLVM_TARGETS_TO_BUILD=Linx

cmake --build compiler/llvm/build-linxisa-clang \
  --target clang lld llvm-objdump llvm-objcopy -j8

export COMPILER_DIR="$LINXISA_ROOT/compiler/llvm/build-linxisa-clang/bin"
"$COMPILER_DIR/clang++" --version
```

The compiler directory must contain `clang`, `clang++`, `ld.lld`,
`llvm-objdump`, and `llvm-objcopy`.

## Build the musl sysroot

```bash
MODE=phase-b bash lib/musl/tools/linx/build_linx64_musl.sh
export LINX_SYSROOT="$LINXISA_ROOT/out/libc/musl/install/phase-b"
test -d "$LINX_SYSROOT/usr/include"
```

The benchmark Makefiles read `LINX_SYSROOT`. Keeping it explicit prevents an
unrelated host sysroot from entering the build.

## Build QEMU

The superproject clean-build helper checks out the pinned QEMU revision in a
detached worktree, configures `linx64-softmmu`, and builds the system emulator:

```bash
export LINX_QEMU_BUILD="$LINXISA_ROOT/out/qemu-linx"
bash tools/bringup/run_qemu_build_clean.sh \
  --qemu-root "$LINXISA_ROOT/emulator/qemu" \
  --out-dir "$LINX_QEMU_BUILD" \
  --target qemu-system-linx64

export QEMU="$LINX_QEMU_BUILD/qemu-system-linx64"
"$QEMU" --version
```

## Verify the environment

```bash
test -x "$COMPILER_DIR/clang++"
test -x "$COMPILER_DIR/llvm-objdump"
test -d "$LINX_SYSROOT"
test -x "$QEMU"

cd "$SUPERNPU_ROOT"
bash docs/examples/check.sh
```

The final command compiles the documentation's C++ and PTO examples. It does
not launch QEMU.

!!! warning "Keep paths external to source"
    Export tool paths in your shell or CI job. Do not commit workstation paths
    to benchmark Makefiles, examples, or documentation.
