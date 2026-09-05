# Lightweight hosted four-PE runtime

The `res_check=on` multi-thread matmul lane links the toolchain-provided static
runtime instead of starting libc on every PE.

- PE0 follows the normal static musl entry and owns file I/O and process exit.
- PE1 through PE3 start at the exported `__linx_group_worker_start` symbol.
- gfrun supplies an independent stack to every PE.
- `linx_group_run()` publishes one single-shot context, directly calls the
  ELF-provided `__linx_group_worker_main`, runs PE0, and
  waits for one completion slot from every PE.
- The functional runtime uses one PE0-written ready bit and one independently
  written completion slot per PE, so it needs no atomic read-modify-write
  helper or indexed shared-state update.
- Workers do not call libc or issue syscalls. PE0 returning from `main()`
  reaches the libc exit path; gfrun treats the leader's `SYS_exit` or
  `SYS_exit_group` as core-wide completion and ends the parked workers.

This first runtime is intentionally static-ELF-only. It is a reference carrier
for the hosted ABI tracked by PTO-ISA/pto-spec issue 150 and
LinxISA/SuperScalarModel issue 346; it is not a pthread implementation.

The runtime is enabled by setting `group_runtime=on` before including
`test/common/Makefile.common`. The multi-thread matmul Makefile does this by
default. The public declaration comes from Linx-TileOP-API and the
implementation comes from the compiler runtime archive installed as
`liblinx_builtin_rt.a`.

The runtime-only four-PE carrier is run with:

```sh
COMPILER_DIR=/path/to/linx-llvm/bin \
MUSL_SYSROOT=/path/to/linx-musl/sysroot \
GFRUN=/path/to/gfrun \
TILEOP_API_ROOT=/path/to/Linx-TileOP-API \
benchmark/one-level-arch/verification/run_hosted_group_runtime_smoke.sh
```
