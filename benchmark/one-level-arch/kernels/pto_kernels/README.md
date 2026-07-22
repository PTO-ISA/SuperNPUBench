# PTO-Kernel Imports

This directory contains tile-only kernels migrated from
[`LinxISA/PTO-Kernel`](https://github.com/LinxISA/PTO-Kernel) revision
`0eb72cb20b0de99326d984a9a27ddb815e6e4c24`.

The migration intentionally includes only sources whose data paths can be
written with the LinxISA PTO 0.57 allowlist. Scalar QEMU fallback branches were
removed. Tile-grid loops remain because they select independent global tensor
tiles; all element and matrix work is performed by PTO intrinsics.

| Family | Migrated kernels | PTO operations |
| --- | --- | --- |
| Memory | `tload_store` | `TLOAD`, `TSTORE` |
| Elementwise | `add_custom` | `TLOAD`, `TADD`, `TSTORE` |
| Matmul | `gemm`, `gemm_basic`, `gemm_demo`, `gemm_performance`, `mamulb`, `tmatmul_acc` | `TLOAD`, `TMATMUL`, `TMATMUL_ACC`, `TCVT`, `TADD`, `TMULS`, `TSTORE` |
| Attention | `flash_attention` | `TLOAD`, `TMATMUL`, `TCVT`, `TADD`, `TSTORE` |

`migration.json` records the exact upstream source path for every import and
pins the isolated support-header copies by SHA-256. Host-simulation code remains
available inside the upstream backend header for syntax testing, but the active
Linx benchmark Makefile must not enable it.
Run `python3 scripts/verify_pto_kernel_migration.py` from the repository root
to verify provenance coverage, wrappers, and intrinsic allowlist compliance.
