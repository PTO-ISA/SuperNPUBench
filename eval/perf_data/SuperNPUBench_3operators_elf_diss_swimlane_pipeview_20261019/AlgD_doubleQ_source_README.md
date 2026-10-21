# AlgoD FP4 FA double-Q trace package

Sq=256; Skv=65536; QD=VD; Tm=128; Tk=256; MXFP4/BF16; double-Q; preload N1.
Model: /Users/blacktraker/Programming/gitproj/DV4/SuperScalarModel-asl (bdbafed55613 + Local B.ASSEMBLE fix).
gfsim SHA256: 6fe43c9f135981a296b515a46475a71649a4227fdaf1eb2f15abbd1f5930bf93.
Zero inputs, no RES_CHECK. MXFP4 measured, not a separately validated HIF4 run. Wakeup diagnostics remain; timing is provisional.
Per-case manifest.json contains exact command, cwd, environment, ELF hash and baseline/trace results.
SwimLane JSON: load in https://ui.perfetto.dev . PipeView .out: load in Konata (full instruction mode).
D192 fails Shared TMA descriptor validation; any trace there is incomplete and must not be used as a completed timeline.

| D | baseline cycles | trace cycles | rc | wakeup |
|---|---:|---:|---:|---:|
| 128 | 143998 | 143998 | 0 | 75 |
| 256 | 285589 | 285589 | 0 | 26 |
| 192 | None | None | -6 | 0 |
| 64 | 143253 | 143253 | 0 | 73 |
