# baremetal gfsim run status

- Run dir: `/Users/liyi/Documents/GitHub/SuperNPUBench/output/run_logs/gfsim_full_baremetal/run_20260625_232108_serial`
- Command style: `gfsim -f <elf>` via PTY, one process at a time.
- Safety change: stopped after serial `gfsim` returned `EXIT_-9`, likely system/resource kill; no gfsim process remains.
- Current baremetal ELF count: 57
- Current matching disassembly count: 57
- Captured full CPU reports with cycles: 3
- Failed/interrupted gfsim rows in summary: 1
- Baremetal compile failures: 8 control ELF(s), linker RAM overflow.

Files:
- `cycle_summary.tsv`: partial gfsim full-run results.
- `cycle_compare_partial.tsv`: partial comparison against previous `gfsim -m 1` summary.
- `compile_failures.tsv`: baremetal compile failures.
- `logs/`: raw per-ELF gfsim logs.
