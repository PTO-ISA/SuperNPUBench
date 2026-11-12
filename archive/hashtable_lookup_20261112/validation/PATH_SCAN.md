# 路径检查记录（NDF §5.1，revision 3）

- 工具：`build/scan_paths.py`，sha256 `54ceb79e7c3452b7461db8fd5a5a62eeed99187a917fc51973c4e63b476b5cd8`
- 命令：`python3 build/scan_paths.py . --jsonl validation/path_scan_revision4.jsonl`
- 范围：归档全部 57 个文件（扫描记录自身除外）；文本按行、二进制按字节偏移；`.gz` 先流式解压再扫描压缩前内容。
- 结果：`files=57 hits=27 forbidden=0`；原始输出见 `path_scan_revision4.log`，逐条见
  `path_scan_revision4.jsonl`（revision 3 的 `path_scan_revision3.*` 作为历史记录保留）。
- 个人绝对路径（`/Users/<name>`、`/home/<name>`、`C:\Users\<name>`、用户名）命中为 0。
- 所有 27 条命中均为 `/tmp` 临时路径（无个人路径）。其中 8 条在结果日志/负控输出中，
  其余在 `validation/*.md` 与 `build_repro_r3.log` 中作为被记录的命令/路径出现。逐项处置如下：

| 文件:行 | 内容 | 判定 | 理由 |
|---|---|---|---|
| `results/r1.../run.log:1` | `/tmp/slots64` | 允许 | gfsim argv 回显，历史原始日志；无个人/源码路径 |
| `results/r2.../run_rw48.log:1` | `/tmp/soa_op` | 允许 | 同上 |
| `results/r2.../run_rw48_way7.log:1` | `/tmp/soa_op` | 允许 | 同上 |
| `results/r2.../run_rw8.log:1` | `/tmp/soa_op` | 允许 | 同上 |
| `validation/negative_control/README.md:26` | `/tmp/does_not_exist.elf` | 允许 | 负控示例，故意指向不存在的文件 |
| `validation/negative_control/missing_elf.log:1` | `/tmp/does_not_exist.elf` | 允许 | 负控原始输出 |
| `validation/negative_control/r1check_bad.log:1` | `/tmp/negctl` | 允许 | 负控临时目录，无个人路径 |
| `results/r3.../README.md:9` | `/tmp/hash_op26` | 已改为 `A-R3-ELF` | 文档占位，见 revision 3 |

- 发布材料（`outputs/hashtable_lookup_archive_assets/release_r3`）另行扫描：
  `files=23 hits=0 forbidden=0`；trace 日志与 meta 已按
  `release_r3/sanitization.json` 的规则脱敏（原件 hash 与派生 hash 均保留）。
- 未解决的个人路径：无。因此本归档的路径检查可按 §5.1 记为通过（仅限上述范围）。

- 新增命中（`validation/PATH_SCAN.md`、`build_repro_r3.log`、`diss_normalization.md`、
  `negative_control/README.md`）均为上文已列路径的重复出现，判定同上：`/tmp` 临时
  路径，非个人/源码路径；构建日志中的 `/tmp/r3_rebuild_*` 是本次重建输出目录。
