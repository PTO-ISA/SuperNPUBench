# repro: gfsim 4-thread SPMD PEID out-of-range

冻结组合（Model #925）：
- SuperNPUBench `codex/pto059-element-if-integration@9076bd0`（本分支基底）
- LinxISA/llvm-project `codex/pto-element-region-compiler@482b4207`
- LinxISA/Linx-TileOP-API `b223de6`
- LinxISA/SuperScalarModel `codex/pto-element-generic-timing@8e8d8ac7`

ELF：`solution_transpose_transpose_simt_PE4.elf`（本目录，含 PEID 读 `ssrget 2050`）

复现：
```bash
# 1) 直接跑（模型 4-PE 自动检测 -> threadCount=4 而 stdPeCount=1）
<model>/bin/gfsim --pto059-execution-mask-dev -f solution_transpose_transpose_simt_PE4.elf
# 抛出: ASSERTION FAILED: writeback.peid < iex_pe_rslv_array.size()  (iex.cpp:586)

# 2) 对照：不读 PEID 的官方用例则正常
<model>/bin/gfsim --pto059-execution-mask-dev -f \
  <bench>/output/kernel/element_wise/tile_element_suite/elf/indexed_gather_tile_element.elf
```
