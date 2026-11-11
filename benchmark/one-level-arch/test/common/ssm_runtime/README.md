# SSM.PTO v1 gfrun runtime

This is the default freestanding platform binding for PLAT=linx with
baremetal=off. There is no runtime=ssm switch. It is not a full musl CRT
or a supported gfsim hosted runtime.

The common Makefile replaces the old startup link with `-static -nostartfiles`
and three objects: adapter, group worker, and the separate assembly main bridge.
The bridge calls existing `int main(void)` once on each PE and returns its status
to the worker. It does not supply argc/argv. Do not link another `_start`, group
worker, or `__linx_group_pe_main` implementation at the same time.

## Pinned upstream sources

- Adapter and header: SuperScalarModel `fb7db9c740e9f9f11c18acd24a0aeb98aa0dea3d`,
  copied unchanged from the runtime documented at
  https://github.com/LinxISA/SuperScalarModel/blob/fb7db9c740e9f9f11c18acd24a0aeb98aa0dea3d/docs/gfrun_hosted_abi.md.
- Worker: LLVM `aac7fcd7d33e4cc6211bce20a284a90b768e4592`, copied unchanged from
  the compiler-rt producer explicitly pinned by that model document:
  https://github.com/LinxISA/llvm-project/blob/aac7fcd7d33e4cc6211bce20a284a90b768e4592/compiler-rt/lib/builtins/linx64v5/group_worker.c.
  This file is not present in the current LLVM branch; the vendored snapshot is
  intentional, not evidence that current musl automatically supplies the worker.

Local adapter:
/Users/blacktraker/Programming/gitproj/DV4/SuperNPUBench/benchmark/one-level-arch/test/common/ssm_runtime/adapter.c

Local worker:
/Users/blacktraker/Programming/gitproj/DV4/SuperNPUBench/benchmark/one-level-arch/test/common/ssm_runtime/group_worker.c

Assembly bridge:
/Users/blacktraker/Programming/gitproj/DV4/SuperNPUBench/benchmark/one-level-arch/test/common/ssm_main_bridge.s

## Operator contract

The model supplies private stacks/TLS and the invocation context. Optional PE0
process/invocation initialization hooks execute before the worker's startup
SYNC.CORE. Shared input initialization belongs in these hooks, not in four
concurrent copies of main. The worker calls main after the startup barrier,
publishes each result, executes the completion barrier, then PE0 requests the
native manager completion; other PEs park. Manager checks precede formal End.

This does not run libc constructors or adapt arbitrary hosted applications.
File I/O requires the model's explicit file-service capability and supported
calls. Existing legacy group dispatch, early exit and custom startup code must
be adapted per operator. CPU and baremetal build paths retain their own startup.

## Completion and existing numerical validation

Worker tail processing already publishes each PE result, performs the final
SYNC.CORE, aggregates results on PE0 and requests manager End/Abort. These are
not process destructors or an application cleanup hook. The adapter currently
only exposes optional process/invocation initialization hooks. Existing main
output handling still runs before the callback returns.

Keep the existing matmul res_check=on / golden-check workflow and
/Users/blacktraker/Programming/gitproj/DV4/SuperNPUBench/benchmark/one-level-arch/test/kernel/matmul/src/gfrun_matmul.py.
No new matmul testcase or numerical checker is installed by this integration.
Historical isolated diagnostic sources and logs are preserved under
/Users/blacktraker/Programming/gitproj/DV4/SuperNPUBench/outputs/ssm_common_shapes_20261111.
Those results do not certify the file-service/RES_CHECK path. File capabilities,
shared initialization and output ordering must be validated through that path.
