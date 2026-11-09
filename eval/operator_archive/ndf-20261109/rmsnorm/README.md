# RMSNorm scheduling snapshots — not numerically accepted

Three preserved header snapshots: `continuous8`, `prefetch_store_priority`, and `dual_output_live`. They change request/prefetch/store scheduling and live ranges; continuous8 retains the strict binary reduction tree. None replaces the main-branch RMSNorm entry in this commit. Use one variant directory as the include root; these headers export the same symbols and cannot be combined in one translation unit.

`driver/rms_norm_dynamic_tree_32k.cpp` is copied byte-for-byte from SuperNPUBench main `c1d9a4f982d218bd14605376fa8c72eb105887c2`, path `benchmark/one-level-arch/test/solution/normalization/rms_norm/src/rms_norm_dynamic_tree_32k.cpp`. It already uses 512B-aligned GM buffers, introduced by `68451bf75414d397c4bcede7a00581ed73585e09`. The old local 128B alignment patch is superseded and is not transplanted. This driver is **current integration support**, not a proven exact source pair for any historical ELF. The continuous8-specific historical driver/build binding was not recovered.

## Historical observations and limits

Old continuous8: 49288 versus same-config baseline 54997 cycles (read/write/shared 128B/c). This is not the current new_dv baseline. In the new_dv resweep (read/write 113B/c, uniform latency 300..400, seed2, CellReg256KiB/PE), continuous8 shared128 is 54423; prefetch-only shared128 is 52404. Dual-output is 51182 with shared0 and 54774 with shared128. These are timing diagnostics; nonzero full numerical validation and negative controls are missing. New_dv continuous8 explicitly reuses the older ELF, SHA256 `b94f1d317c12717f348bf3248cb58dc0bf328fbfbf7edce94fdfdf3cf88409ae`; it does not introduce another continuous8 source version.

Snapshot names for the other two candidates preserve historical directory labels. Their source-to-ELF byte identity has not been reproduced by this source-only submission; do not infer that identity from a directory name or the cycle table.

Response queue depth, observer/bandwidth changes, L2 bank/port scheduling, ideal bank bypass and store-source staging are **model** changes and remain outside this operator snapshot. Source/runtime/effective-config identity and nonzero correctness must be closed before making a performance claim. The old continuous8 native config is missing; the runtime currently at its old path differs from its historical hash.

## Dependencies and reproduction gaps

Headers depend on `common/pto_tileop.hpp` and the matching Linx TileOP/ISA/compiler contract. The driver also needs existing NPUBench `test/common` headers/runtime and the solution test build flags (`G_A`, `G_R`, `DType`, `PE_NUM=4`). Functional builds need real input/gamma files, RES_CHECK/ENABLE_BINARY_OUTPUT and a portable CHK_DIR. Default test shapes are [128,8192] and [32,16384]; the 32768 reduction experiment uses explicit shape flags.

This commit ships sources only: no ELF/diss, input generator/golden, native/effective configs or traces. No build/run reproduction was performed. Rebuild with the registered dependency lock, add independent nonzero golden/readbacks/negative controls, and collect paired SwimLane runs before promotion. Missing artifacts cannot count as PASS.
