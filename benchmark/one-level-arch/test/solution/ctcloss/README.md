# Ctcloss_v2 network test

This follows the normalization test layout:

- Makefile includes ../../common/Makefile.common.
- compile.all builds ctcloss_v2_single_pe and ctcloss_v2_four_pe.
- src/ctcloss_v2_*_pe.cpp are testcase entries sharing ctcloss_v2_test.hpp.
- src/gen_ctcloss_v2_data.py writes ordinary input/golden/tiling BINs.
- src/ctcloss_v2_data_compare.py compares RES_CHECK output files.
- src/validate_ctcloss_v2_inputs.cpp validates the full input and tiling ABI.
- validate.py is optional orchestration around these Makefile/script entries;
  it has no separate compiler flag or startup implementation.

Only the requested network specification is present:
log_probs float32 [195,256,41]; targets int32 [256,74];
input_lengths int32 [256] = [195]*256;
target_lengths int32 [256] = [40]+[74]*255.
blank=0, reduction=none, zero_infinity=false.
Outputs are loss float32 [256] and log_alpha float32 [256,195,149].
Sample 0 has 81 valid states; its remaining states are -Inf.

The deterministic synthetic fixture uses seed 20261024 / NumPy PCG64.
Logits are normalized to FP32 log-softmax; effective labels are 1..40.
This is not captured network data or a comparison against ops-nn device output.
The full FP64 time-first reference is independently cross-checked with scalar
FP64 DP for samples 0,1,255. The six raw input/golden files remain byte-identical
to the earlier network fixture. Only PE-specific tiling metadata differs.

## Standard entries

Requires Python 3.11+, NumPy, host g++, and the pinned Linx dependencies.
No packages are installed automatically. Set:
~~~sh
export COMPILER_DIR=/path/to/linx_blockisa_llvm_musl/bin
export API_INCLUDE=/path/to/Linx-TileOP-API/include
cd benchmark/one-level-arch/test/solution/ctcloss
~~~

One normal RES_CHECK case:
~~~sh
python3 src/gen_ctcloss_v2_data.py --out-dir /tmp/ctc-pe4-data --pe-num 4
make -o clean -W "$PWD/src/ctcloss_v2_four_pe.cpp" \
  TESTCASE=ctcloss_v2_four_pe DType=float PE_NUM=4 res_check=on \
  CHK_DIR=/tmp/ctc-pe4-data diss
# After a successful four-PE gfrun execution of the generated ELF:
python3 src/ctcloss_v2_data_compare.py --cmp-dir /tmp/ctc-pe4-data
~~~

For single PE select ctcloss_v2_single_pe, PE_NUM=1 and --pe-num 1.
The generator requires a new output directory. CHK_DIR defaults to
compare/<ELF-basename>, following the common Makefile convention; when using
that default, generate the inputs in that exact directory.

compile.all builds both PE variants. If CTC_COMPARE_ROOT is set, it uses
<root>/pe1/compare and <root>/pe4/compare for their CHK_DIR directories.
Set res_check=off for the timing variants. ELF names encode DType, the fixed
network dimensions, PE count and res_check mode to prevent artifact confusion.
The shared Makefile contains an implicit clean prerequisite; compile.all
suppresses it with -o clean and forces the relevant TU for flag/CHK_DIR changes.
Kernel/header dependencies are also tracked by this testcase Makefile.
The public common Makefile itself is unchanged.

Optional full generation/host checking/four-configuration build:
~~~sh
python3 validate.py --build-only \
  --compiler-dir "$COMPILER_DIR" --api-include "$API_INCLUDE" \
  --model /path/to/SuperScalarModel
~~~

Remove --build-only to run gfrun accuracy then real-L2 gfsim timing;
--functional-only skips timing. Limits remain 120 seconds / 900 seconds.
Timeouts fail; this fixed network shape is never reduced. A fresh run directory
contains commands, return codes, source/tool/ELF hashes, dependency HEAD/dirty
state, Make plans, disassembly, and logs. Runtime output directories are fresh,
so stale output files cannot qualify as a PASS.

## Runtime / RES_CHECK

Both modes use static hosted musl startup and the existing common
group_worker_runtime.o. There is no testcase-private start.s or input .incbin.
The frozen loader gives PE0 the libc entry and PE1..3 the common worker entry,
with independent stacks. Workers return to the common park loop; only PE0
performs file I/O or exits the process. One-PE execution uses the same ABI
without worker participants.

PE0 reads all four input BINs plus tiling_info.bin, validates signed values and
the tiling ABI, and publishes inputs using MultiThreadResCheckSync. That common
protocol is used unchanged; it is not a newly implemented hardware barrier.
The host computes ownership/strides in tiling_info; the kernel applies them.
After the kernel, PE0 waits for four completion slots before exporting results.

CTC lengths/labels determine executed work. Unlike a zero-input timing stub,
res_check=off still reads the same inputs before TRACE.begin, so it cannot
silently benchmark zero effective lengths. RES_CHECK controls guard setup and
result export. File I/O and launch setup lie outside TRACE.begin/TRACE.end.
The validator reports gfsim Total Cycles as whole-program cycles, including
startup/I/O; it does not turn them into a kernel-only speedup measurement.

RES_CHECK writes loss.bin [256], log_alpha.bin [256,195,149], status.bin and
output_guards.bin. The comparator requires exact PE IDs/completion/sample
counts and exact guard bytes, rejects NaNs, matches Inf signs, and uses
rtol=1e-4 / atol=2e-5 for finite numbers. It does not read simulator memory
dumps. One PE must own 256 samples; each of four PEs must own 64.

## Dependency and validation status

- LLVM codex/pto-element-region-compiler:
  482b4207e88adf312fd1b1a8a567b83f5a4e207b.
- TileOP: b223de6a80c1d9b7c2f1ac3a1ff4517b1951d650.
- Toolchain wrapper: e6a31efb4cfb17f1f1c33265cbf6dbb61bbba156.
- Model codex/pto-element-generic-timing:
  8e8d8ac7b1becc8937e77989c2762139f946b8af plus local GPR scalar-owner routing
  backport (DispatchToPE(stid)); not pristine frozen code or latest main.
  That model patch is not included in this kernel PR.

Validation on 2026-10-25: host input/tiling ABI, full FP64 reference checks,
and single/four-PE RES_CHECK on/off builds PASS. Checker acceptance/rejection
self-tests use the complete network shape and are not target-kernel runs.
Full-network gfrun/gfsim: NOT_RUN; no verified operator performance.
Related historical blockers:
https://github.com/LinxISA/SuperScalarModel/issues/1075
https://github.com/LinxISA/SuperScalarModel/issues/982
Upstream routing fix:
https://github.com/LinxISA/SuperScalarModel/pull/1000
