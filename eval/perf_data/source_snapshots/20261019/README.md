# Performance workbook implementation sources (2026-10-19)

This directory publishes function-body sources for review without integrating
experimental changes into the normal benchmark build. `sources.json` records
the original absolute path, function line, and SHA256 of every copied file.

- `matmul_lowp_reuseA.hpp`, `matmul_lowp.cpp`, and
  `Makefile.matmul_fullKcache` are the retained full-K-cache build snapshots.
  The header SHA256 is
  `5686c4ac33a67663fa6b152bcc8e7f50e59565fa1868a7003859bd6c34556d98`.
- `fa_lowp_algD.hpp` is the exact AlgoD double-Q / preload-N1 snapshot used by
  the comparison. Its SHA256 is
  `6fdf25b4d182e87e2b9699a24d81b273514baf591bdb68efc64d43bf2027ecc8`.
  The comparison uses MXFP4, not a separately validated HiF4 implementation;
  completed runs have wakeup warnings and D192 fails an assertion.
- Other headers are review snapshots of available local implementations, or
  of the already-published generic implementation where identified by `HEAD:`.
  They are not proof of source-to-binary identity for retained historical ELFs.
  In particular, the standard FA timing runs reused existing ELFs.
- The best FP16 M/K/N=288/7392/256 reuseB ELF remains available, but its
  `matmul_reuseB_mtail_cellA` implementation is missing. No substitute header
  is claimed to reproduce that binary. The available shared-A variant is a
  different implementation, even though its source is retained here.

This is a source-review archive, not a self-contained replacement toolchain or
simulator. The compiler's PTO headers and benchmark harness are still required.
Workbook metrics, run flags, warnings, and original evidence remain unchanged.
