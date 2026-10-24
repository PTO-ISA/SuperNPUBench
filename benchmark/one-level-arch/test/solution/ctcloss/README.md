# Ctcloss_v2: one fixed network specification, single PE and four PE

This directory intentionally contains one network fixture. There is no small
shape / smoke / boundary-case suite in this change.

Inputs, in order:
- log_probs: float32 [195,256,41], contiguous [T,N,C].
- targets: int32 [256,74], contiguous padded labels.
- input_lengths: int32 [256], exactly [195]*256.
- target_lengths: int32 [256], exactly [40]+[74]*255.

Attributes: blank=0, reduction=none, zero_infinity=false.
Outputs: loss float32 [256], log_alpha float32 [256,195,149].
Sample 0 has 81 valid states; unused alpha states are defined as -Inf.
Targets beyond sample 0's first 40 labels are padding (filled with zero).

## Data, reference and checks

generate.py creates deterministic synthetic data (seed 20261024, NumPy PCG64).
Random logits are normalized to FP32 log-softmax; effective labels are in 1..40.
This is shape-aligned synthetic input, not captured network data or an ops-nn
implementation-to-implementation comparison.

The complete alpha/loss reference uses time-first FP64 DP. Samples 0,1,255 are
also checked with an independent scalar FP64 implementation. FP32 raw logical
goldens are log_alpha.golden.bin / negative_log_likelihood.golden.bin.
alpha.golden and loss.golden additionally contain four guard words at each end.
The checker compares guard bytes exactly, finite values with rtol=1e-4 and
atol=2e-5, rejects NaNs and compares Inf signs. Each real PE must publish its
identity, completion and sample count (64 per PE for four PE).

validate_inputs.cpp validates these full-size INT32 inputs using the adapter;
it does not run scalar CTC as a replacement for the target kernel.

## Build and execution

Requires Python 3.11+, NumPy, host g++, and the compatible Linx compiler/runtime,
public TileOP headers and model. No dependencies are installed automatically.

From the repository root (replace the three dependency paths):
~~~bash
python3 benchmark/one-level-arch/test/solution/ctcloss/run.py \
  --toolchain /path/to/linx_blockisa_llvm_musl \
  --tileop-include /path/to/Linx-TileOP-API/include \
  --model /path/to/SuperScalarModel \
  --build-only
~~~

By default both PE variants are built from the same generated input/data
object. Use --pe-count 1 or --pe-count 4 to select one. Remove --build-only to
run target checks; --functional-only skips gfsim. Existing timeout limits are
120 seconds for gfrun and 900 seconds for real-L2 gfsim. This full shape may
exceed those limits: a timeout fails, does not reduce the shape, and never
becomes an accuracy PASS. gfsim runs only after gfrun accuracy succeeds.

Default output is a unique validation_logs/ctcloss-solution-* directory beside
the repository, overridable by --out. It includes commands, return codes,
timings, source/tool/ELF/input hashes, repo HEAD/dirty state, IR, disassembly,
network-spec.json and runner logs/dumps. Large binaries/data are generated
outside the checkout and are not included in this PR.

The direct-boot start.s relies on the loader supplying independent per-PE
stacks and joining all exits before the host checks outputs. Guard buffers
are initialized in the ELF image, so workers cannot race a leader's memset.
This is not a hosted libc or hardware device launcher.

## Frozen dependency tuple and validation limit

- LLVM: codex/pto-element-region-compiler @
  482b4207e88adf312fd1b1a8a567b83f5a4e207b.
- TileOP: b223de6a80c1d9b7c2f1ac3a1ff4517b1951d650.
- Toolchain wrapper: e6a31efb4cfb17f1f1c33265cbf6dbb61bbba156.
- Model: codex/pto-element-generic-timing @
  8e8d8ac7b1becc8937e77989c2762139f946b8af, plus a local scalar-owner routing
  backport: VecTop's GPR packet uses DispatchToPE(stid), not vector coreIndex.
  This is not pristine frozen code and not latest model main. The backport is
  not part of this kernel PR; upstream handling is described in model PR #1000.

The runner records actual source dirty hashes and installed runner hashes.
Validation on 2026-10-24: full input dimensions/dtypes/lengths, FP64 reference
cross-checks and host INT32 adapter validation PASS; single-PE and four-PE target
compilation/linking and element-region lowering checks PASS.
Full-network gfrun/gfsim: NOT_RUN. There is no verified performance result.
Related historical model failures:
https://github.com/LinxISA/SuperScalarModel/issues/1075
https://github.com/LinxISA/SuperScalarModel/issues/982
