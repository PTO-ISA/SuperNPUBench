# HIF4 reuseA refresh (2026-10-22)

The two HIF4 matmul artifact directories were rebuilt from the current
`matmul_lowp_reuseA` source and refreshed together with their timing traces.
The spreadsheet reports the operator interval (`Group Total Cycles`), while
each `*.gfsim.log` also retains the full-program `Total Cycles`.

| M/K/N | tile M/N/K | resident N outputs | Group cycles | effective MFU |
|---|---:|---:|---:|---:|
| 48/7168/96 | 128/32/2048 | 3 | 4,784 | 21.0702% |
| 256/4096/1280 | 128/128/512 | 2 | 46,593 | 87.9102% |

Compiler checkout:
`/Users/blacktraker/Programming/gitproj/DV4/linx-toolchain-build` at
`e6a31efb4cfb17f1f1c33265cbf6dbb61bbba156`.

Timing model state:
`/Users/blacktraker/Programming/gitproj/DV4/SuperScalarModel-asl`, base commit
`bdbafed55613955d0b8b27438e9e7d2a72dfccd2`, plus the saved three-source-file
Cube/block scheduling delta in
[HIF4_MODEL_STATE_bdbafed_plus_ad8490782.patch](https://github.com/PTO-ISA/SuperNPUBench/blob/main/eval/perf_data/SuperNPUBench_3operators_elf_diss_swimlane_pipeview_20261019/HIF4_MODEL_STATE_bdbafed_plus_ad8490782.patch).
The rebuilt `gfsim` SHA256 is
`b423fef9a5ff6419360e304027da2fc58d430499c27515803a150b963217e6c1`.

The exact gfsim command is recorded as the first line of each `*.gfsim.log`.
Both runs use SoC bandwidth 156 B/cycle, mean latency 260 cycles, SharedTReg
bandwidth 256 B/cycle, 256 KiB/64 SSB SharedTReg capacity, internal
accumulators, and `fourpe.conf`.
