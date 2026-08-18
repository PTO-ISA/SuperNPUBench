# fixp microbenchmark report

Source: `microbenchmark/fixp/src/fixp_tmatmul.cpp` (one binary per fixp configuration), built and disassembled by `microbenchmark/fixp/compile.all`.

Kernel: a single call to one of the 12 matrix post-process operations (TMATMUL{,_BIAS,_ACC,_MX,_MX_BIAS,_MX_ACC} + TGEMV{,_BIAS,_ACC,_MX,_MX_BIAS,_MX_ACC}) using the new-style `OP(dst, a, b, ..., fixp::Options)` interface. A/B are FP16 Left/Right tiles; C/Bias/Scale are FP32/FP16 aux tiles; the destination carries the converted result. Toolchain: Linx BlockISA LLVM 15.0.4 (linx64v5-musl), `linx-toolchain-build-latest`.

## Build

Requires the Linx BlockISA toolchain on PATH via `COMPILER_DIR` (default `PLAT=linx`). From this directory:

```bash
# 0. point at the latest worktree toolchain
export COMPILER_DIR=/path/to/linx-toolchain-build-latest/output/linx_blockisa_llvm_musl/bin

# 1. build + disassemble one variant
make FIXP_MODE=S_QF_S8 diss        # -> output/.../fixp_tmatmul_s_qf_s8_*.elf{.diss}
make FIXP_MODE=BIAS diss          # TMATMUL_BIAS variant
make FIXP_MODE=GEMV diss          # TGEMV variant

# 2. build + disassemble all 64 variants (prints PASS/FAIL table)
bash compile.all                 # log: compile.fixp.log

# 3. regenerate this report from the .diss files
python3 report_fixp.py           # -> fixp_report.md
```

Tile shape defaults `M=N=K=TM=TN=TK=32` (override with `M=... TM=...`); `FIXP_MODE` is upper-cased to a `-D` define and lower-cased for the ELF suffix. Mode labels are short (`bias`/`acc`/`mx`/`gemv`/...) so the `-D` macro never collides with the op function name. The `diss` target runs `llvm-objdump -dl` to emit the `.elf.diss` next to each `.elf`.

Result summary: **PASS=63 FAIL=0 BLOCKED=1 (total=64)**

## Coverage vs doc (12 operations x B.FPATR options)

All 12 documented operations share one B.FPATR options mechanism (PreQuantMode x scalar/vector quant x ReLU/LReLU/PReLU x RowMax/GroupMax/MaxAbs). The TMATMUL modes sweep the full FPATR parameter face; the other 11 operations are each exercised with a parameter-free call plus (where useful) an s8 scalar-quant call to verify the options overload emits the right IOR stream.

| operation | param-free mode(s) | options mode(s) | notes |
| --- | --- | --- | --- |
| TMATMUL | keep_acc (+f16/bf16/relu, 43 FPATR-sweep modes) | s_qf_s8 / v_qf_s8 / s8_lrelu / vqf_s8_prelu / ... | Shared (shared, s8_shared); legacy3 |
| TMATMUL.BIAS | bias | bias_s8 |  |
| TMATMUL.ACC | acc | acc_s8 |  |
| TMATMULMX | mx | mx_s8 |  |
| TMATMULMX.BIAS | mxbias | - |  |
| TMATMULMX.ACC | mxacc | - |  |
| TGEMV | gemv | gemv_s8 | vec=Left(1xK), mtx=Right(KxN), M=1 |
| TGEMV.BIAS | gemv_bias | - |  |
| TGEMV.ACC | gemv_acc | - |  |
| TGEMVMX | gemv_mx | gemv_mx_s8 |  |
| TGEMVMX.BIAS | gemv_mx_bias | - |  |
| TGEMVMX.ACC | gemv_mx_acc | - |  |

## B.FPATR attribute decode

`B.FPATR PreQuant, Relu, GroupNCode, RowMaxEn, GroupMaxEn, RowMaxInit, MaxAbsEn`; encoding word `0x2023 | PreQuant<<26 | Relu<<23 | GroupNCode<<19 | RowMaxEn<<18 | GroupMaxEn<<17 | RowMaxInit<<16 | MaxAbsEn<<15`.

| mode | op | shared | option chain / call | expected FPATR | actual word | GPR (B.IOR) | aux IOT | status | detail |
| --- | --- | :---: | --- | --- | --- | ---: | ---: | --- | --- |
| keep_acc | TMATMUL | no | `fixp::keep_acc()` | 0, 0, 0, 0, 0, 0, 0 | 0x00002023 | 0 | 0 | PASS | mnemonic + FPATR + encoding + IOR + IOS + aux match |
| keep_acc_relu | TMATMUL | no | `keep_acc().relu()` | 0, 1, 0, 0, 0, 0, 0 | 0x00802023 | 0 | 0 | PASS | mnemonic + FPATR + encoding + IOR + IOS + aux match |
| f16 | TMATMUL | no | `fixp::f16()` | 1, 0, 0, 0, 0, 0, 0 | 0x04002023 | 0 | 0 | PASS | mnemonic + FPATR + encoding + IOR + IOS + aux match |
| f16_relu | TMATMUL | no | `f16().relu()` | 1, 1, 0, 0, 0, 0, 0 | 0x04802023 | 0 | 0 | PASS | mnemonic + FPATR + encoding + IOR + IOS + aux match |
| bf16 | TMATMUL | no | `fixp::bf16()` | 16, 0, 0, 0, 0, 0, 0 | 0x40002023 | 0 | 0 | PASS | mnemonic + FPATR + encoding + IOR + IOS + aux match |
| bf16_relu | TMATMUL | no | `bf16().relu()` | 16, 1, 0, 0, 0, 0, 0 | 0x40802023 | 0 | 0 | PASS | mnemonic + FPATR + encoding + IOR + IOS + aux match |
| s_reqs8 | TMATMUL | no | `scalar<REQS8Pre>(desc)` | 3, 0, 0, 0, 0, 0, 0 | 0x0c002023 | 1 | 0 | PASS | mnemonic + FPATR + encoding + IOR + IOS + aux match |
| s_deqf16 | TMATMUL | no | `scalar<DEQF16>(desc)` | 5, 0, 0, 0, 0, 0, 0 | 0x14002023 | 1 | 0 | PASS | mnemonic + FPATR + encoding + IOR + IOS + aux match |
| s_shifts16 | TMATMUL | no | `scalar<SHIFTS322S16>(desc)` | 13, 0, 0, 0, 0, 0, 0 | 0x34002023 | 1 | 0 | PASS | mnemonic + FPATR + encoding + IOR + IOS + aux match |
| s_qf_s4 | TMATMUL | no | `scalar<QF322S4Pre>(desc)` | 17, 0, 0, 0, 0, 0, 0 | 0x44002023 | 1 | 0 | PASS | mnemonic + FPATR + encoding + IOR + IOS + aux match |
| s_qf_s16 | TMATMUL | no | `scalar<QF322S16Pre>(desc)` | 19, 0, 0, 0, 0, 0, 0 | 0x4c002023 | 1 | 0 | PASS | mnemonic + FPATR + encoding + IOR + IOS + aux match |
| s_qf_s8 | TMATMUL | no | `s8(desc) / scalar<QF322S8Pre>` | 24, 0, 0, 0, 0, 0, 0 | 0x60002023 | 1 | 0 | PASS | mnemonic + FPATR + encoding + IOR + IOS + aux match |
| s_qf_hif8 | TMATMUL | no | `scalar<QF322HIF8Pre>(desc)` | 25, 0, 0, 0, 0, 0, 0 | 0x64002023 | 1 | 0 | PASS | mnemonic + FPATR + encoding + IOR + IOS + aux match |
| s_qf_fp8 | TMATMUL | no | `scalar<QF322FP8Pre>(desc)` | 26, 0, 0, 0, 0, 0, 0 | 0x68002023 | 1 | 0 | PASS | mnemonic + FPATR + encoding + IOR + IOS + aux match |
| s_qf_f32 | TMATMUL | no | `scalar<QF322F32Pre>(desc)` | 27, 0, 0, 0, 0, 0, 0 | 0x6c002023 | 1 | 0 | PASS | mnemonic + FPATR + encoding + IOR + IOS + aux match |
| s_qf_f16 | TMATMUL | no | `scalar<QF322F16Pre>(desc)` | 32, 0, 0, 0, 0, 0, 0 | 0x80002023 | 1 | 0 | PASS | mnemonic + FPATR + encoding + IOR + IOS + aux match |
| s_qf_bf16 | TMATMUL | no | `scalar<QF322BF16Pre>(desc)` | 34, 0, 0, 0, 0, 0, 0 | 0x88002023 | 1 | 0 | PASS | mnemonic + FPATR + encoding + IOR + IOS + aux match |
| s_qs_bf16 | TMATMUL | no | `scalar<QS322BF16Pre>(desc)` | 35, 0, 0, 0, 0, 0, 0 | 0x8c002023 | 1 | 0 | PASS | mnemonic + FPATR + encoding + IOR + IOS + aux match |
| v_reqs8 | TMATMUL | no | `vector<VREQS8Pre>(tile)` | 2, 0, 0, 0, 0, 0, 0 | 0x08002023 | 0 | 1 | PASS | mnemonic + FPATR + encoding + IOR + IOS + aux match |
| v_deqf16 | TMATMUL | no | `vector<VDEQF16>(tile)` | 4, 0, 0, 0, 0, 0, 0 | 0x10002023 | 0 | 1 | PASS | mnemonic + FPATR + encoding + IOR + IOS + aux match |
| v_shifts16 | TMATMUL | no | `vector<VSHIFTS322S16>(tile)` | 12, 0, 0, 0, 0, 0, 0 | 0x30002023 | 0 | 1 | PASS | mnemonic + FPATR + encoding + IOR + IOS + aux match |
| v_qf_s4 | TMATMUL | no | `vector<VQF322S4Pre>(tile)` | 18, 0, 0, 0, 0, 0, 0 | 0x48002023 | 0 | 1 | PASS | mnemonic + FPATR + encoding + IOR + IOS + aux match |
| v_qf_s16 | TMATMUL | no | `vector<VQF322S16Pre>(tile)` | 20, 0, 0, 0, 0, 0, 0 | 0x50002023 | 0 | 1 | PASS | mnemonic + FPATR + encoding + IOR + IOS + aux match |
| v_qf_s8 | TMATMUL | no | `s8(quant_tile) / vector<VQF322S8Pre>` | 23, 0, 0, 0, 0, 0, 0 | 0x5c002023 | 0 | 1 | PASS | mnemonic + FPATR + encoding + IOR + IOS + aux match |
| v_qf_hif8 | TMATMUL | no | `vector<VQF322HIF8Pre>(tile)` | 28, 0, 0, 0, 0, 0, 0 | 0x70002023 | 0 | 1 | PASS | mnemonic + FPATR + encoding + IOR + IOS + aux match |
| v_qf_f16 | TMATMUL | no | `vector<VQF322F16Pre>(tile)` | 33, 0, 0, 0, 0, 0, 0 | 0x84002023 | 0 | 1 | PASS | mnemonic + FPATR + encoding + IOR + IOS + aux match |
| v_qf_bf16 | TMATMUL | no | `vector<VQF322BF16Pre>(tile)` | 36, 0, 0, 0, 0, 0, 0 | 0x90002023 | 0 | 1 | PASS | mnemonic + FPATR + encoding + IOR + IOS + aux match |
| v_qf_fp8 | TMATMUL | no | `vector<VQF322FP8Pre>(tile)` | 37, 0, 0, 0, 0, 0, 0 | 0x94002023 | 0 | 1 | PASS | mnemonic + FPATR + encoding + IOR + IOS + aux match |
| v_qf_f32 | TMATMUL | no | `vector<VQF322F32Pre>(tile)` | 38, 0, 0, 0, 0, 0, 0 | 0x98002023 | 0 | 1 | PASS | mnemonic + FPATR + encoding + IOR + IOS + aux match |
| v_qs_bf16 | TMATMUL | no | `vector<VQS322BF16Pre>(tile)` | 39, 0, 0, 0, 0, 0, 0 | 0x9c002023 | 0 | 1 | PASS | mnemonic + FPATR + encoding + IOR + IOS + aux match |
| s8_relu | TMATMUL | no | `s8(desc).relu()` | 24, 1, 0, 0, 0, 0, 0 | 0x60802023 | 1 | 0 | PASS | mnemonic + FPATR + encoding + IOR + IOS + aux match |
| s8_lrelu | TMATMUL | no | `s8(desc).lrelu(fp19)` | 24, 2, 0, 0, 0, 0, 0 | 0x61002023 | 2 | 0 | PASS | mnemonic + FPATR + encoding + IOR + IOS + aux match |
| v_s8_relu | TMATMUL | no | `s8(quant).relu()` | 23, 1, 0, 0, 0, 0, 0 | 0x5c802023 | 0 | 1 | PASS | mnemonic + FPATR + encoding + IOR + IOS + aux match |
| f16_prelu | TMATMUL | no | `f16().prelu(fp19_tile)` | 1, 3, 0, 0, 0, 0, 0 | 0x05802023 | 0 | 1 | PASS | mnemonic + FPATR + encoding + IOR + IOS + aux match |
| s8_prelu | TMATMUL | no | `s8(desc).prelu(fp19_tile)` | 24, 3, 0, 0, 0, 0, 0 | 0x61802023 | 1 | 1 | PASS | mnemonic + FPATR + encoding + IOR + IOS + aux match |
| rowmax | TMATMUL | no | `keep_acc().row_max(out)` | 0, 0, 0, 1, 0, 0, 0 | 0x00042023 | 0 | 0 | PASS | mnemonic + FPATR + encoding + IOR + IOS + aux match |
| rowmax_init | TMATMUL | no | `keep_acc().row_max(in,out)` | 0, 0, 0, 1, 0, 1, 0 | 0x00052023 | 0 | 1 | PASS | mnemonic + FPATR + encoding + IOR + IOS + aux match |
| groupmax_8 | TMATMUL | no | `keep_acc().group_max<8>(out)` | 0, 0, 1, 0, 1, 0, 0 | 0x000a2023 | 0 | 0 | PASS | mnemonic + FPATR + encoding + IOR + IOS + aux match |
| groupmax_16 | TMATMUL | no | `keep_acc().group_max<16>(out)` | 0, 0, 2, 0, 1, 0, 0 | 0x00122023 | 0 | 0 | PASS | mnemonic + FPATR + encoding + IOR + IOS + aux match |
| groupmax_128 | TMATMUL | no | `keep_acc().group_max<128>(out)` | 0, 0, 9, 0, 1, 0, 0 | 0x004a2023 | 0 | 0 | PASS | mnemonic + FPATR + encoding + IOR + IOS + aux match |
| rowgroup_maxabs | TMATMUL | no | `keep_acc().row_max(in,out).group_max<8>(out).max_abs()` | 0, 0, 1, 1, 1, 1, 1 | 0x000fa023 | 0 | 1 | PASS | mnemonic + FPATR + encoding + IOR + IOS + aux match |
| f16_groupmax | TMATMUL | no | `f16().group_max<16>(out)` | 1, 0, 2, 0, 1, 0, 0 | 0x04122023 | 0 | 0 | PASS | mnemonic + FPATR + encoding + IOR + IOS + aux match |
| s8_rowmax | TMATMUL | no | `s8(desc).row_max(out)` | 24, 0, 0, 1, 0, 0, 0 | 0x60042023 | 1 | 0 | PASS | mnemonic + FPATR + encoding + IOR + IOS + aux match |
| bias | TMATMUL.BIAS | no | `TMATMUL_BIAS(c,a,b,bias,keep_acc())` | 0, 0, 0, 0, 0, 0, 0 | 0x00002023 | 0 | 0 | PASS | mnemonic + FPATR + encoding + IOR + IOS + aux match |
| acc | TMATMUL.ACC | no | `TMATMUL_ACC(d,c,a,b,keep_acc())` | 0, 0, 0, 0, 0, 0, 0 | 0x00002023 | 0 | 0 | PASS | mnemonic + FPATR + encoding + IOR + IOS + aux match |
| mx | TMATMULMX | no | `TMATMUL_MX(c,a,sa,b,sb,keep_acc())` | 0, 0, 0, 0, 0, 0, 0 | 0x00002023 | 0 | 0 | PASS | mnemonic + FPATR + encoding + IOR + IOS + aux match |
| mxbias | TMATMULMX.BIAS | no | `TMATMUL_MX_BIAS(d,a,sa,b,sb,bias,keep_acc())` | 0, 0, 0, 0, 0, 0, 0 | 0x00002023 | 0 | 0 | PASS | mnemonic + FPATR + encoding + IOR + IOS + aux match |
| mxacc | TMATMULMX.ACC | no | `TMATMUL_MX_ACC(d,c,a,sa,b,sb,keep_acc())` | 0, 0, 0, 0, 0, 0, 0 | 0x00002023 | 0 | 0 | PASS | mnemonic + FPATR + encoding + IOR + IOS + aux match |
| gemv | TGEMV | no | `TGEMV(d,mtx,vec,keep_acc())` | 0, 0, 0, 0, 0, 0, 0 | 0x00002023 | 0 | 0 | PASS | mnemonic + FPATR + encoding + IOR + IOS + aux match |
| gemv_bias | TGEMV.BIAS | no | `TGEMV_BIAS(d,mtx,vec,bias,keep_acc())` | 0, 0, 0, 0, 0, 0, 0 | 0x00002023 | 0 | 0 | PASS | mnemonic + FPATR + encoding + IOR + IOS + aux match |
| gemv_acc | TGEMV.ACC | no | `TGEMV_ACC(d,c,mtx,vec,keep_acc())` | 0, 0, 0, 0, 0, 0, 0 | 0x00002023 | 0 | 0 | PASS | mnemonic + FPATR + encoding + IOR + IOS + aux match |
| gemv_mx | TGEMVMX | no | `TGEMV_MX(d,mtx,smtx,vec,svec,keep_acc())` | 0, 0, 0, 0, 0, 0, 0 | 0x00002023 | 0 | 0 | PASS | mnemonic + FPATR + encoding + IOR + IOS + aux match |
| gemv_mx_bias | TGEMVMX.BIAS | no | `TGEMV_MX_BIAS(d,mtx,smtx,vec,svec,bias,keep_acc())` | 0, 0, 0, 0, 0, 0, 0 | 0x00002023 | 0 | 0 | PASS | mnemonic + FPATR + encoding + IOR + IOS + aux match |
| gemv_mx_acc | TGEMVMX.ACC | no | `TGEMV_MX_ACC(d,c,mtx,smtx,vec,svec,keep_acc())` | 0, 0, 0, 0, 0, 0, 0 | 0x00002023 | 0 | 0 | PASS | mnemonic + FPATR + encoding + IOR + IOS + aux match |
| bias_s8 | TMATMUL.BIAS | no | `TMATMUL_BIAS + s8(desc)` | 24, 0, 0, 0, 0, 0, 0 | 0x60002023 | 1 | 0 | PASS | mnemonic + FPATR + encoding + IOR + IOS + aux match |
| acc_s8 | TMATMUL.ACC | no | `TMATMUL_ACC + s8(desc)` | 24, 0, 0, 0, 0, 0, 0 | 0x60002023 | 1 | 0 | PASS | mnemonic + FPATR + encoding + IOR + IOS + aux match |
| mx_s8 | TMATMULMX | no | `TMATMUL_MX + s8(desc)` | 24, 0, 0, 0, 0, 0, 0 | 0x60002023 | 1 | 0 | PASS | mnemonic + FPATR + encoding + IOR + IOS + aux match |
| gemv_s8 | TGEMV | no | `TGEMV + s8(desc)` | 24, 0, 0, 0, 0, 0, 0 | 0x60002023 | 1 | 0 | PASS | mnemonic + FPATR + encoding + IOR + IOS + aux match |
| gemv_mx_s8 | TGEMVMX | no | `TGEMV_MX + s8(desc)` | 24, 0, 0, 0, 0, 0, 0 | 0x60002023 | 1 | 0 | PASS | mnemonic + FPATR + encoding + IOR + IOS + aux match |
| shared | TMATMUL | yes | `TMATMUL(d,a,SharedTile<B>,keep_acc())` | 0, 0, 0, 0, 0, 0, 0 | 0x00002023 | 0 | 0 | PASS | mnemonic + FPATR + encoding + IOR + IOS + aux match |
| s8_shared | TMATMUL | yes | `TMATMUL + SharedTile<B> + s8(desc)` | 24, 0, 0, 0, 0, 0, 0 | 0x60002023 | 1 | 0 | PASS | mnemonic + FPATR + encoding + IOR + IOS + aux match |
| lrelu_only | TMATMUL | no | `keep_acc().lrelu(fp19)  [zero,lrelu]` | 0, 2, 0, 0, 0, 0, 0 | - | 1 | 0 | BLOCKED | no .diss (assembler rejected bundle) |
| vqf_s8_prelu | TMATMUL | no | `s8(quant_tile).prelu(prelu_tile)` | 23, 3, 0, 0, 0, 0, 0 | 0x5d802023 | 0 | 1 | PASS | mnemonic + FPATR + encoding + IOR + IOS + aux match |
| legacy3 | TMATMUL | no | `TMATMUL(c,a,b) legacy 3-param` | 0, 0, 0, 0, 0, 0, 0 | 0x00002023 | 0 | 0 | PASS | mnemonic + FPATR + encoding + IOR + IOS + aux match |

## Notes

- **Scalar quant modes** pass the descriptor through a single `B.IOR [quant],[]` GPR slot; `s8_lrelu` additionally loads the FP19 LReLU slope as `B.IOR [quant,lrelu],[]` (two real GPRs). The `zero` placeholder in `B.IOR [zero,...]` is never counted as a real GPR.
- **LReLU without scalar quant** (`lrelu_only`, gap C) is BLOCKED: the C++ template dispatches IorMode==2 and emits `B.IOR [zero,%LReluGpr],[]`, but the LLVM backend cannot match that instruction pattern (Match Instruction Error), so no .diss is produced. The mode documents this toolchain gap.
- **Vector quant / PReLU modes** add extra `B.IOT` source lines after the math operands (quant or PReLU parameter Tile, valid 1 x N). `vqf_s8_prelu` chains vector-quant + PReLU (SrcMask=6); the two aux tiles share a single source line, so aux counts 1 source line (the FPATR fields already encode the quant+prelu combination).
- **Operation-family modes** (BIAS/ACC/MX/GEMV) verify the BSTART.CUBE mnemonic and math operand stream: ACC adds a C accumulator tile, BIAS adds a Bias tile, MX adds ScaleA/ScaleB (TMATMULMX / TGEMVMX, spelled without a dot). The aux count is derived as `iot_src - min(MATH_SRC[(mnemonic, shared)], iot_src)` so the math operand lines (1-3 per op) are not mistaken for PostProcess aux lines.
- **Shared-Right** (`shared`/`s8_shared`) wraps B in `SharedTile<RightTile>`, lowering it via `B.IOS` instead of a `B.IOT` source line. Only the TMATMUL family accepts Shared operands; TGEMV is Local-only.
- **Legacy 3-param** (`legacy3`, `TMATMUL(c,a,b)` no options) emits A/B as read operands on the destination line (combined read-write `B.IOT ... ->reg`), so it has 0 math source lines; the `min` clamp keeps aux at 0 instead of -1. Its FPATR is the default (0,0,0,0,0,0,0), same as `keep_acc`.
- **RowMax / GroupMax / MaxAbs** (8 modes) now assemble and PASS with the latest worktree toolchain (commit 86959776); the intermediate non-final destination `B.IOT ... ->reg` that PTO-ISA v0.58 requires is now accepted. RowMaxIn contributes 1 aux source line (`rowmax_init`, `rowgroup_maxabs`); RowOut / GroupOut are destinations (not counted).
- GroupN code mapping used: 8->1, 16->2, 32->3, 48->4, 64->5, 80->6, 96->7, 112->8, 128->9.
