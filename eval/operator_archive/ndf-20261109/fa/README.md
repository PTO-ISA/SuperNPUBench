# FA AlgoD2 scoped packed rowsum — experimental source snapshot

This is the final identified 8K performance candidate, **not a numerically accepted operator**. The source is preserved byte-for-byte; it is not wired into the default kernel test suite. AlgoD already exists at `benchmark/one-level-arch/kernels/basic_op/fa/fa_lowp_algD.hpp` (introduced by `757d4e598bebfd0bbd1d4c47dddd8371bd15410c`); that main-branch implementation is distinct from this exploration baseline.

## Source and arithmetic

`source/multi_thread/fa_lowp_algB/fa_lowp_algD2.hpp` is the scoped final candidate; `fa_lowp_algD.hpp` is its immediate exploration baseline. `source/fa_lowp_algD_driver.cpp` is the original SPMD driver. D2 compresses the denominator update per query/KV block from 31 to 15 core TileOPs: 8 TROWSUM + 4 TPACK + TMUL + TROWSUM + TMULS. Pair-local temporaries shorten live ranges. The D-to-D2 snapshot delta also moves V load after softmax; it must not be described as only changing rowsum.

The implementation approximates unscaled attention using BF16 log references on an ln2 grid, MXFP4/E8M0 probabilities and FP32 PV accumulation. The denominator uses BF16 exponentials, while the numerator uses quantized probabilities. It is not an exact FP32 softmax implementation; packed reduction also changes rounding order.

## Frozen evidence (historical, not rerun for this commit)

Sq=256, Skv=8192, qD=vD=128, Tm=128, Tk=256, 4 PEs, K preload=1, N=1, dblQ=1, QK preload=0: AlgoD 21488 versus scoped D2 20519 cycles (-4.51%; reported MFU 79.85%). At Skv=65536 the comparison regresses: 155269 versus 258099 cycles. These are shape/config-specific observations, not an accepted performance claim.

Packed rowsum smoke and its negative control passed. Full attention later failed: 6756/32768 outputs exceed atol=1e-3, rtol=2e-2; old/new compiler full outputs are byte-identical. All six input readbacks matched and output size was 65536 bytes. The historical BroadcastByteOffset model issue is only a clue; causality is unresolved. Compiler #140 passing does not close full numerical acceptance.

## Build dependencies and remaining work

Use the original driver with `-I source` and the matching NPUBench `kernels` and `test/common` includes. Its intended flags are `-DFA_ALGD2=1 -DFA_ALGB_ALGO_D=1 -DFA_ALGB_D_PRELOAD=1 -DFA_ALGB_D_PRELOAD_N=1 -DFA_ALGB_D_DBLQ=1 -DFA_ALGB_D_PRELOAD_QK=0 -DTsq=256 -DTskv=8192 -DFA_QD=128 -DFA_VD=128 -DTm=128 -DTk=256`; functional builds additionally require RES_CHECK/ENABLE_BINARY_OUTPUT, a portable CHK_DIR, and the existing multi-thread worker runtime. Debug builds need shape macros visible before the header.

The archived API transplant is in `dependencies/api_tpack_emit.patch`; it belongs to Linx-TileOP-API, not to the NPUBench kernel. It preserves Local MIDDLE/LAST parent lifetime and subview offsets. Do not blindly apply it to a different API revision. Historical toolchain/runtime identities are in `identity.json`; the exact API/ISA baseline and native effective configuration remain incomplete. The registered newer compiler previously rejected full descriptors/unroll forms; this snapshot does not claim to build on current main.

No ELF, input files, golden or trace is published by this source-only commit. Full independent oracle/input generation, negative controls, portable build/test integration, current-compiler admission, native/effective configs and new 8K/64K SwimLane evidence must be completed before promotion to a default kernel. Missing artifacts or dependencies must be treated as failures, not successful skips.
