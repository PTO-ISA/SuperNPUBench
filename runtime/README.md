# Host–device benchmark runtime

The native C++ host owns a gfrun process for the lifetime of a benchmark.
Each device hardware thread runs the common resident firmware from
SuperScalarModel. The first consumer is the [one-level TADD benchmark](../benchmark/one-level-arch/test/kernel/vec/host_device).

## Build and run

Use a POSIX host with a C++17 compiler, a Linx V5 toolchain with the TileOP
math headers and sysroot, a Linx-TileOP-API checkout providing
`common/pto_launch.hpp` and `common/pto_message.hpp`, and a SuperScalarModel
checkout with a built `bin/gfrun` supporting `--hscb-host`.
Run from the SuperNPUBench repository root.

```sh
export COMPILER_DIR=/path/to/linx_blockisa_llvm_musl/bin
export TILEOP_API_ROOT=/path/to/Linx-TileOP-API
export MODEL_ROOT=/path/to/SuperScalarModel
./compile_all.sh host-device
make -C benchmark/one-level-arch/test/kernel/vec/host_device run
```

The build links the model's resident MsgBuffer worker with the benchmark's
Linx kernel in one ELF. It writes a function-symbol index next to the ELF.
Use the index from that exact build. The host executable, ELF and index are in
the benchmark's `output/` directory. Set `OUTPUT` to a fresh directory when
changing the toolchain or dependency revisions. `HOST_CXX` selects the native
compiler and `GFRUN` overrides the simulator path.

## Add a benchmark

1. Write a native host `main()` and a separate Linx kernel with the C signature
   `extern "C" uint64_t kernel_name(uint64_t argument)`. The return value is
   zero for success. Share the argument layout in a header with fixed-width
   fields. Addresses in that structure are guest GM addresses.
2. Construct `supernpu::Device(gfrun, firmware)`, use `allocate` to reserve
   GM, and `upload` to copy input and the argument structure. `initialize()`
   releases the workers and waits until each has reported READY and parked.
3. Call `launch("kernel_name", blockDim, argumentAddress)`. In the kernel use
   `pto::GetBlockIdx()` and `pto::GetBlockNum()` to split work. For independent
   tiles, block `i` visits tile `i`, `i + blockDim`, and so on. Each block
   returns through the firmware, which posts DONE and waits for more work.
4. Call `wait()` before reading, updating inputs, or starting another launch.
   A nonzero kernel status raises an exception. Use `download` to copy results
   to native memory. `close()` checks the model exit status and releases the
   session allocations.
5. Set `HOST_MAIN`, `HOST_HEADERS` and `DEVICE_KERNEL` in the case Makefile and
   include [device/host_device.mk](device/host_device.mk).

The backend serializes launches and transfers. Transfers use temporary binary
files on the local host. HSCB carries the entry PC, argument address and
dispatch identity. Each independent block may run on any resident hardware
thread. Operators containing a fixed-participant hardware barrier require a
compatible worker group and launch policy before migration.

The TADD host checks 17 tiles of 64 FP32 elements with 8 logical blocks, uses
the first result as the second input, then repeats with 4 blocks. Each round
checks every element, each active block's register values, and the untouched
geometry entries for inactive blocks.
