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
Linx kernel in one ELF. In live host mode gfrun reads the function symbols from
the ELF it loaded. A kernel handle belongs to that session. Build outputs are
in the benchmark's `output/` directory. Set `OUTPUT` to a fresh directory
when changing toolchain or dependency revisions. `HOST_CXX` selects the native
compiler and `GFRUN` overrides the simulator path. To exercise queue and
lifetime behavior with the native host tests, run
`make -C benchmark/one-level-arch/test/kernel/vec/host_device test-host`.

## Submit a kernel chain

One host submission thread owns each device session and its default FIFO stream. The TADD host allocates its
buffers before boot, then calls `initialize()` to run the workers to READY and
BWE. The host queues the following work before its only `synchronize()`:

1. Upload the three inputs and initialize the two block-geometry records.
2. Submit TADD with eight logical blocks to compute `tmp = a + b` in GM.
3. Reuse the same host argument object and kernel handle. Submit TADD with four
   blocks to compute `out = tmp + c` in GM.
4. Download the final output and the two records.

The host compares all 1,088 FP32 elements with `(a + b) + c`. It checks the
logical block IDs and counts from both launches, releases the temporary buffer
and verifies that a new allocation at the same address has a distinct lifetime.

## Add a benchmark

Write a native `main()` and one or more Linx entries with the C signature
`extern "C" uint64_t kernel_name(uint64_t argument)`. Returning zero indicates
success. Use the [generic kernel launch ABI](KERNEL_LAUNCH_ABI.md) for new
kernels. Set `HOST_MAIN`, `HOST_HEADERS` and `DEVICE_KERNEL_SOURCES` in the case
Makefile and include [device/host_device.mk](device/host_device.mk).

- **Buffers.** Use `allocateBuffer(bytes, alignment)` to obtain a session-owned
  handle. Allocation and release are synchronous operations. Get a GM address for an argument field with `device.address(buffer)`.
  Use `release(buffer)` to make the allocation available for reuse after earlier work completes. Copies of
  the released handle become invalid even if a new allocation has the same GM
  address. Initialize a reused allocation before reading it.
- **Kernels and arguments.** Resolve each entry once with
  `device.kernel("kernel_name")`, then reuse the handle across launches. Pass
  the block count and a `KernelCall<Schema>` to `enqueueLaunch`. Buffer arguments
  record their GM addresses and retain their allocations automatically. Fixed-
  width scalar arguments and the complete descriptor are snapshotted on
  submission and retained through matching DONE.
- **Copies and synchronization.** `enqueueUpload` snapshots the host input.
  Call the enqueue methods after `initialize()`. `enqueueDownload` needs a
  writable host destination that remains alive until
  `synchronize()` completes. The stream executes operations in submission order;
  a later kernel sees the writes of earlier kernels. Synchronization reports
  execution errors. Explicit `close()` checks the model's exit status.
- **Block geometry.** In a kernel, `pto::GetBlockIdx()` identifies the current
  logical block and `pto::GetBlockNum()` returns the submitted count. For
  independent tiles, block `i` visits tile `i`, `i + blockDim`, and so on.
  Operators with a fixed-participant hardware barrier also need a compatible
  worker group and dispatch policy.

The example ELF exports two distinct entries. `bench_tadd` consumes four buffer
arguments and a 64-bit tile count. `bench_fill` consumes a buffer, 32-bit count,
signed value and a zero 64-bit step. A separate negative run confirms that a
schema mismatch returns a device error before the implementation runs.

The background host thread drives one gfrun command at a time. Transfers use
temporary binary files. This backend models ordering, completion and data
visibility for a single stream. Its host elapsed time includes IPC and file I/O.
