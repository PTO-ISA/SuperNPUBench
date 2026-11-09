# RMSNorm single-buffer timing snapshot, 2026-11-09

This separate source archive preserves the historical **43,694 Total Cycles**
baseline. It adds no default operator or test entry and does not replace the
IO ping-pong kernel merged in [PR271](https://github.com/PTO-ISA/SuperNPUBench/pull/271).
It contains no continuous8, prefetch-store-priority or dual-output variant.

## Frozen source

- Kernel: SuperNPUBench `e7a4a761b880b9ea820e5106b64756ebbc6da998`, identical
  to the parent of IO ping-pong commit `92ecf231270dcfdbca975580d9644c7e68eb0ed6`.
- Driver: the same historical driver, with only input/gamma/output declarations
  changed to `alignas(512)`. The R=8192 path does not use partial_workspace;
  its declaration remains historical to preserve the measured source bytes.
- Alignment reference: wangyu's already merged
  [PR267](https://github.com/PTO-ISA/SuperNPUBench/pull/267), commit
  `68451bf75414d397c4bcede7a00581ed73585e09`. Current main also aligns workspace.
- Support headers and startup assembly come from the same historical commit.

Complete source and historical artifact hashes are in [identity.json](identity.json).
No personal ELF paths or unpublished input files are required by this zero-input
build. Input and gamma are static zero buffers, A=128/R=8192/FP16/4PE;
RES_CHECK is deliberately disabled. No nonzero correctness check or negative
control has been performed; this is a scheduling observation, not acceptance.

## Build and optional timing reproduction

From any working directory, set `ARCHIVE_DIR` to this directory and run:

```bash
python3 "$ARCHIVE_DIR/verify_sources.py"
python3 "$ARCHIVE_DIR/build_snapshot.py" \
  --toolchain-bin "$LINX_TOOLCHAIN_BIN" --outdir "$NEW_RUN_DIR"
```

The build requires the frozen clang identified by SHA256 in identity.json,
with its matching TileOP API headers, sysroot and linker. It emits an ELF,
matching disassembly, build logs and command/hash manifest to a new output
directory outside this archive. Prefix maps remove build-source/toolchain paths.
It refuses to overwrite an output directory or silently substitute a compiler.

To also run timing, append `--gfsim "$HISTORICAL_GFSIM"` to the build command.
The supplied runtime must match the recorded historical model hash. The runner
copies the native configuration into the output directory's `configs/`, uses
that directory as cwd for the old configuration parser, and sets
`RMS_TAIL_BYPASS=1`. Python 3 and Linux `resource` support are required; the
SwimLane helper uses a 6GiB host address-space ceiling for JSON export.
It produces `perf_run/rmsnorm_swim.log`, `rmsnorm.json` and
`rmsnorm_nocounters.json` and fails on simulator errors or missing/invalid trace.

The historical timing conditions are PR1145 lazy allocation/first-data-ready,
tail bypass, 256KiB CellReg/PE, read/write 128 B/c each with aggregate 128 B/c,
random SoC uniform[300,400], seed=2. The native conf is shipped byte-for-byte.
The helper and streaming JSON reader preserve instruction/flow events while
filtering PMU counters from the nocounters file.

The Git source archive keeps small metadata and source files. The final ELF/diss and compiler-free
runner are published separately as the operator Release attachment. The matching portable model
runtime and optional historical traces are in the model Release; no compiler download is needed
to replay the existing ELF. Source rebuilding remains optional and has the historical tool limits above.

## Scope of the result

The same fixed historical runtime/config measured this single-buffer version at
43,694 cycles and the exploratory dual-output version at 44,246 cycles.
No valid same-configuration comparison against current main's IO ping-pong
version exists: it hit a descriptor assertion with that historical tool
combination. This archive does not claim IO ping-pong is slower or propose
reverting it. The PR271-reported improvement belongs to its own configuration.

The archived recipe was also rebuilt from a copied source directory and run
from an unrelated cwd: **43,694 Total Cycles**, rc=0, with full/nocounters
SwimLane produced. The portable ELF instruction listing matches the historical
baseline; ELF and disassembly personal-path scans pass. This check does not add
nonzero numerical validation.


## Release 下载复跑

[SuperNPUBench Release](https://github.com/PTO-ISA/SuperNPUBench/releases/tag/work-archive-20261109)
提供 `rmsnorm-single-buffer-43694-operator-r1.tar.gz`（28,489 bytes，ELF/diss、源码快照、免编译runner）。
[Model Release](https://github.com/LinxISA/SuperScalarModel/releases/tag/work-archive-20261109)
提供 `model-rmsnorm-6467f7f4-runtime-r1.tar.gz`（8,247,968 bytes，runtime、运行库/loader、配置及模型问题映射）。
两包解压在同级目录后：

```bash
python3 rmsnorm-operator-r1/replay.py --model-root rmsnorm-runtime-r1 --outdir replay_run
```

Linux x86_64、Python>=3.9。其他cwd请使用实际解压路径；输出目录必须为新目录且在payload之外。
不需要编译器、完整工具链或历史trace。两份可选附件为历史trace及固定commit+patch的模型源码重建recipe。
具体URL、大小、hash和输入合同见 [artifact_delivery.json](artifact_delivery.json)；Release还附跨仓库
`release_index.json` 与 `download_verification.json`。

四附件均已实际从GitHub下载并校验；下载后的算子/runtime在干净环境、无原作者源码/build/run目录依赖下
复跑 **43694 Total Cycles / rc=0**，新full/nocounters trace hash与历史结果一致。
新runtime SHA6467f7f4来自同一base+8文件patch的prefix-map源码重建，原981546a2因包含个人路径不分发；
新旧身份分别记录，没有覆盖历史二进制或修改二进制字符串。它只通过这份零输入计时case，尚无非零数值验收。

算子源码固定commit仍为 `877aa052dfc6dc2c6cf1493a76d08b1472cc3b12`，文档/Release tag不替代运行身份。
默认IO ping-pong保持；最终同ELF未修改主线模型对照、独立S复核及长期维护人仍缺。
模型改动/issue关系见 [SuperScalarModel PR1221](https://github.com/LinxISA/SuperScalarModel/pull/1221)。
