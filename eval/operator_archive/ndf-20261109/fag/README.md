# FAG direct backward — experimental source snapshots

`compact_source` preserves compact/colexpand/subfirst for Sq1024; `prefetch_source` preserves the Sq4096 prefetch-V1 candidate with runtime output scales. Each contains `basic_op/fa/fa_gmma_bwd_q.hpp` and the matching `fa_gmma_bwd_q.cpp` driver. They export the same operator and must be built separately. They are not connected to the default test suite by this archive commit.

The FP8 E4M3 inputs are Q/K/V/dO; lse, delta and outputs dQ/dK/dV are FP32. The kernel loops over Q inside a KV tile, uses MatDType P/dS operands and FP32 accumulation, compact auxiliary vectors and colexpand subtraction, dV internalACC, dQ atomic accumulation, and optional direct CUBE-to-Shared transport. dQ must be initialized before accumulation. Prefetch V1 overlaps next-Q/auxiliary loads and reuses the next operands; runtime output scaling remains explicit. Header comments and macros preserve exact implementation assumptions.

## Historical evidence, kept separate

| Candidate/config | Cycles | Validation scope |
|---|---:|---|
| compact, Sq1024/Skv128/D256, 4PE, stream-only 001 | 29028 | Historical nonzero exact evidence is separate from zero-input timing/ablation; not a same-run performance acceptance |
| prefetch V1, Sq4096/Skv128/D256, SL2 data slots512 | 89646 | Fixed historical ELF; equivalence to predecessor does not prove independent golden correctness |
| same prefetch ELF, SL2 data slots768 | 88050 | Hardware-resource diagnostic (+32KiB); no second kernel change |

The Sq4096 predecessor passed only 0.005-tolerance comparison; dQ/dK were not bit-exact. Prefetch/output-scale equivalence to that predecessor must not be upgraded to independent exact-golden validation. The output-scale-only experiment was slower (93188 versus 92787). The compact and prefetch timings use different shapes/model assumptions and cannot be compared as a single optimization series.

## Existing public provenance

SuperScalarModel commit `6156d025568eb2cc8c95225da9fc2f84cd64a022` contains the output-scale predecessor sources at `perf_handoff/fag_internalacc_output_scale_lastuse_20261103/source/`. Commit `516b9d4b77265eafeca087c6303e7ec1af5d8906` contains its handoff artifacts; `8f5321d56a5336afba1211ef358dbf8540ed7179` adds the prefetch patch/runner/result. These commits belong to `explore/tmov-tlsu-cell-static-20261103`; they are absent from workshop archive commit `2547509024c1ed8ff00b0a89a27d8107f637fa9a`. They were not previously SuperNPUBench operator commits.

Prefetch ELF SHA256 is `c7364cfdf4de0f0a31ff8d09fad3079811bb6ed707529d40186dff3a9fe5b552`, shared by 89646/88050. Historical prefetch runtime source is Model `59d965a55d5e90dbc6652d9bb1f6d9590941a46d`, gfsim SHA256 `6a6acc4ae305f000899e3190e7c2a26034528a19afb6fa34d0232a954dbaba8d`. Compact integrated model identity is `eb9079d4c1597f99da4058af7ba87dedc322dbe4`, gfsim SHA256 `ddc9c9eb86fdb871d6aedef7739d06824f7314b19491f558cb3024772d1cadb8`. Model transport/streaming flags and capacity changes are not kernel changes and are not applied here.

## Dependencies and remaining gaps

Build one variant with its directory as the include root, matching TileOP headers plus NPUBench `kernels`, `test/common` and `test/common/src` support. Drivers need benchmark/fileop/multi_thread_res_check headers. Use explicit Sq/Skv/D/Tk and direct-shared/compact-aux settings; the default driver Skv=256 does not select the reported Skv128 case. Functional builds require RES_CHECK/ENABLE_BINARY_OUTPUT, the group worker runtime and portable CHK_DIR/FA_BWD_CHECK_DIR. Inputs are srcq/srck/srcv/srcdo plus src{lse,delta} for compact aux (or expanded lseB/deltaB); prefetch additionally reads three runtime output scales. All expected readbacks and all three outputs must be checked.

This source-only submission includes no new ELF, generated inputs/golden, config export or trace and performs no new numerical/performance run. Compiler/API/ISA complete locks, exact flags, input generation, fresh independent golden/negative controls, source-to-ELF reconstruction, same-run nonzero timing and independent review remain required. Use the public Model handoff as provenance, not as evidence that this snapshot is independently reproducible on current main. Missing inputs/dependencies are failures, never successful skips.
