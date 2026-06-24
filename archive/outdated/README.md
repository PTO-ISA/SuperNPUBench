# Outdated Archive

This directory preserves superseded or unusable material that should not be the default benchmark navigation surface. Nothing here was deleted; the table below records why each item moved and where active work should happen instead.

| Archived path | Rationale | Replacement |
| --- | --- | --- |
| [`tests/other/tileop_api`](tests/other/tileop_api) | Legacy duplicate TileOP API surface with committed generated logs under `script/checknum_true`. | [`../../benchmarks/api/tileop`](../../benchmarks/api/tileop) |
| [`tests/other/py_api`](tests/other/py_api) | Older Python API duplicate. Active Python correctness material is kept outside the benchmark tree. | [`../../tests/py_api`](../../tests/py_api) |
| [`tests/accelerator/v220`](tests/accelerator/v220) | Superseded legacy NPU validation surface, not part of the active Linx benchmark catalog. | [`../../benchmarks/npu`](../../benchmarks/npu) |
| [`tests/accelerator/v310`](tests/accelerator/v310) | Superseded legacy NPU validation surface, not part of the active Linx benchmark catalog. | [`../../benchmarks/npu`](../../benchmarks/npu) |
| [`compiler/linx_blockisa_llvm_musl.tar.gz`](compiler/linx_blockisa_llvm_musl.tar.gz) | Checked-out file is a Git LFS pointer, not a usable compiler archive. | Provide a real Linx compiler path via `COMPILER_DIR`. |

Archive files may retain historical path references because they document the old layout. Do not add new active benchmark cases here.
