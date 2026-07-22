# Build the Toolchain

The supported setup uses this repository inside its parent source tree so the
compiler, sysroot, simulator, headers, and benchmark harness stay on matching
revisions.

## Prerequisites

Install Git, CMake, Ninja, Python 3, a host C/C++ compiler, and the build
dependencies required by LLVM and QEMU. Then clone the parent tree with
submodules:

```bash
git clone --recurse-submodules https://github.com/LinxISA/linx-isa.git
cd linx-isa
git submodule sync --recursive
git submodule update --init --recursive
export LINXISA_ROOT="$PWD"
export SUPERNPU_ROOT="$LINXISA_ROOT/workloads/SuperNPUBench"
```

The names above are literal repository and environment names. The programming
model pages use generic block/thread and NPU terminology.

## Build the Target Compiler

Configure the in-tree LLVM fork for the target backend used by the benchmarks:

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

`COMPILER_DIR` must contain `clang`, `clang++`, `ld.lld`, `llvm-objdump`, and
`llvm-objcopy`.

## Build the Sysroot

```bash
MODE=phase-b bash lib/musl/tools/linx/build_linx64_musl.sh
export LINX_SYSROOT="$LINXISA_ROOT/out/libc/musl/install/phase-b"
test -d "$LINX_SYSROOT/usr/include"
```

Benchmark Makefiles read `LINX_SYSROOT`. Keeping it explicit prevents a host
sysroot from entering target builds.

## Build the Simulator

```bash
export LINX_QEMU_BUILD="$LINXISA_ROOT/out/qemu-linx"
bash tools/bringup/run_qemu_build_clean.sh \
  --qemu-root "$LINXISA_ROOT/emulator/qemu" \
  --out-dir "$LINX_QEMU_BUILD" \
  --target qemu-system-linx64

export QEMU="$LINX_QEMU_BUILD/qemu-system-linx64"
"$QEMU" --version
```

## Verify the Environment

```bash
test -x "$COMPILER_DIR/clang++"
test -x "$COMPILER_DIR/llvm-objdump"
test -d "$LINX_SYSROOT"
test -x "$QEMU"

cd "$SUPERNPU_ROOT"
bash docs/examples/check.sh
```

The final command compiles the documentation examples. It does not launch the
simulator.

!!! warning "Keep Paths External"
    Export tool paths in your shell or CI job. Do not commit workstation paths
    to Makefiles, examples, or documentation.
