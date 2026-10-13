# Tk256 performance experiment

Sq=256, Skv=8192, Tm=128, qD=vD=128, AlgoD dblQ, no preload/constKV. Archived gfsim and same fusion/BW256/internalACC overrides. User requested no further soc_random runs. Random attempt started before steering had already failed; it is excluded from this comparison.

| Tk | Total Cycles | Matmul theory cycles | Matmul-based MFU |
|---|---:|---:|---:|
|128, current source rebuilt|23995|16384|68.28%|
|256, isolated guard relaxation|23096|16384|70.94%|

Delta=-899 cycles (-3.7466%), speedup=1.038924x. MFU gain=2.6578 percentage points. Full kernel measurement including setup, vector, memory and synchronization. Logical MAC=536870912, aggregate peak=32768 MAC/cycle. Tk128: 64 KV blocks, 256 global matmuls; Tk256: 32 KV blocks, 128 global matmuls. Each Tk256 matmul contributes128 theory cycles; Tk128 contributes64.

Archived Tk128=24269 cycles, current rebuilt Tk128=23995. ELF .text hashes differ, so Tk-only comparison uses current rebuilt source pair, not archived ELF.

## Observed counters

| Metric | Tk128 | Tk256 | Scope |
|---|---:|---:|---|
|CMD issued slots|60616|58007|all threads aggregate|
|Internal ACC reads|4032|1984|CUBE0|
|L0A resident misses|354|192|CUBE0|
|L0 in-flight attributed cycles|2679|1294|CUBE0 exclusive category|
|Store in-flight attributed cycles|1170|1398|CUBE0 exclusive category|
|No-tile attributed cycles|1476|1576|CUBE0 exclusive category|
|CellReg store bytes|2129920|2129920|CUBE0|
|CellReg average utilization|30.51%|48.91%|STID0|

L0 wait and inter-block accumulation traffic decrease. Store wait and occupancy increase. CMD slots decline only4.30%, not50%: doubling Tk expands group-wise softmax work per iteration. These are observations, not independent additive speedup estimates; no bandwidth sensitivity experiment was run. Residual bound is mixed compute/writeback/operand/control dependency.

## Build and validation

Original fa_lowp_algD.hpp rejects Tk256 with static assertion. Experimental copy in experiment_include permits128 or256; original source unchanged. Both builds -j4, same defines exceptTk, matching nonempty .elf and .elf.diss. Disassembly inspected for TLOAD shapes, compiler TMOV and internalACC CCTRL; no TMOV rewriting performed. Both simulator runs exit0 and produce full and nocounters SwimLane. Performance builds use res_check=off; this is not a numerical-correctness validation for Tk256.

ELFs, compiler commands, simulator command/overrides/hashes: manifest.json and build_tk128.log/build_tk256.log. Run directories: tk128_rebuilt_baseline/ and baseline_no_soc_random/. Each includesrun_swim.log,run.json,run_nocounters.json,helper.log,command.txt and configured wrapper.

Tk256 ELF: /data/xxr/feishu/workspace/superscalar_v5/performance_archive/2026-10-11_dblQ_final_47p48MFU/cube_tlsu_combined/tmov_chain_fix_20261012/tk256_compare_20261013/build_tk256/kernel/multi_thread/fa_lowp_algB/elf/kernel_multi_thread_fa_lowp_algB_D1_dblQ_C0_GM2_Sq256_Skv8192_Tk256_checkoff.elf
Tk128 ELF: /data/xxr/feishu/workspace/superscalar_v5/performance_archive/2026-10-11_dblQ_final_47p48MFU/cube_tlsu_combined/tmov_chain_fix_20261012/tk256_compare_20261013/build_tk128/kernel/multi_thread/fa_lowp_algB/elf/kernel_multi_thread_fa_lowp_algB_D1_dblQ_C0_GM2_Sq256_Skv8192_Tk128_checkoff.elf
