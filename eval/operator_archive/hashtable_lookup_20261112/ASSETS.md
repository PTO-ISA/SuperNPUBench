# 外部资产清单（NDF §3.7.4）

状态：`uploaded` = 已发布到 Release；`download_verified` = 已从 Release 下载并核对
字节数/SHA256（2026-11-12，见 `validation/material_check.md`）；`uploaded（git）` =
已随归档提交、无需 Release。

Release：https://github.com/PTO-ISA/SuperNPUBench/releases/tag/hashlk-20261112-r3
（固定 commit `4359f297`）。下载前缀：
`https://github.com/PTO-ISA/SuperNPUBench/releases/download/hashlk-20261112-r3/`

| asset_id | 用途 | Release 文件 | 字节 | SHA256 | 状态 |
|---|---|---|---:|---|---|
| `A-COMPILER-PATCH` | required_for: build_reproduced | 归档内 `dependencies/compiler_llvm_pr146/compiler.patch` | 46,475 | `5fef1a21a823c37e251b8fcc904837da25efd9546e9b6f4c3b9d691210553a5d` | uploaded（git） |
| `A-REF-ELF` | required_for: dataset_generation, replay | `reference_hash_6144_PE4.elf` | 40,886,656 | `960ba161b8c662a74bf3f93ed5ad96a75f3375d4a91eee4607bc870b9fa48bd4` | uploaded + download_verified |
| `A-R1-ELF` | required_for: replay_existing_elf（r1） | `r1_64l_2p2_perf.elf` | 61,306,432 | `2324ea4c438bb2eb3b6214190611c2c643b06338c348784a59d0301ef2e3b7d6` | uploaded + download_verified |
| `A-R2-ELF` | required_for: replay_existing_elf（r2） | `r2_256l_2p6_rw48_perf.elf` | 61,306,304 | `35218ed00c5f34cd2a30d0c8ef7fea66ecfd5047611575d397246813383f1e2f` | uploaded + download_verified |
| `A-R3-ELF` | optional_for: compile_evidence | `r3_operator_adaptive_perf.elf` | 40,860,136 | `38a5f34af9b0ab65d690d4c2766319aef5bde991f72933a4a5fa31933214476b` | uploaded + download_verified |
| `A-R1-TRACE` | required_for: perf_trace（r1） | `r1_trace_r3.tar.gz` | 16,155,225 | `7909adc0b77f8bd68c3ae33d7c2b870e259c122f73c70c630167c42a99a7d2ea` | uploaded + download_verified |
| `A-R2-TRACE` | required_for: perf_trace（r2 rw48） | `r2_trace_r3.tar.gz` | 8,156,283 | `2bcbcb0be0d025a1cefb5342aec5b5a4f87513973df9ffb5436369a3251c7805` | uploaded + download_verified |
| `A-R2-WAY7-TRACE` | required_for: perf_trace（r2 way7） | `r2_way7_trace_r3.tar.gz` | 7,928,317 | `0425fad91919009bfe3689220de2427238701e225d448ade25d894748b187aa9` | uploaded + download_verified |
| `A-R2-RW8-TRACE` | required_for: perf_trace（r2 rw8） | `r2_rw8_trace_r3.tar.gz` | 8,491,639 | `11855369754368f4a4a28be3df880379fc1ecbdb9385e4ee68e479a0eadf3e82` | uploaded + download_verified |
| `A-PIPEV` | required_for: pipeview_review | `pipeview_r3.tar.gz` | 22,623,778 | `d43aadd65bf193b041479ae9a15feec913570ead11125fbc6cd84417eb8c1f5a` | uploaded + download_verified |
| `A-API-SNAPSHOT` | required_for: build_reproduced | `api_snapshot_r3.tar.gz`（169 文件，解压后 1,863,758 B） | 207,959 | `a84075b812b0d49fe27ee456f1a06b00655e3db9e4a91df745779942c8518c9e` | uploaded + download_verified |
| `A-ARCHIVE-R3` | 历史快照（revision 3，旧路径） | `hashlk_archive_r3.tar.gz` | 212,367 | `3a61008aba0d0dac1184030e051fcf249f9839e9a7a6afcdef27bca1b04d2fa2` | uploaded + download_verified（历史） |
| `A-ARCHIVE-R5` | required_for: material_check（当前路径） | `hashlk_archive_r5.tar.gz` | 220,886 | `470923ca70b30ea6488c858b475f472c94fdc9b9fbc50c2b41bb357100d82441` | uploaded + download_verified |

## 组合与使用

- 重放 r1：`A-R1-ELF` + 模型 runtime（`dependencies/simulator_pr1177`）+ `config/random300.json`
  + 输入（`dependencies/dataset`）；trace 仅用于审阅。
- 重放 r2：`A-R2-ELF` + 同一模型 + `config/random300.json` + `sl2.RWDB_ENTRY_COUNT=48`。
- 重建：`A-COMPILER-PATCH` + 上游 `llvm-project@a75987bcc` + `A-API-SNAPSHOT` + 数据/参考 ELF。
- trace 包内容：sanitized `swim.log` + `meta.json` + 完整 `swim.json.gz` +
  `swim_nocounters.json.gz`；解压命令 `gzip -dc <file>.gz`。

## 脱敏与打包说明

- `swim.log` / `meta.json` 中的个人路径替换为 `<REPO>` / `<GFSIM_WORKTREE>` /
  `<R2_ELF_DIR>`；JSON trace 内容未改。原件/派生 hash 见
  `outputs/hashtable_lookup_archive_assets/release_r3/sanitization.json` 和
  `validation/PATH_SCAN.md`。
- tar 包使用 `--uid 0 --gid 0 --uname ndf --gname ndf` 打包，避免把本机用户名写入
  tar header；首次上传的旧包已作废，当前 hash 以上表为准。
- 下载验证记录：`validation/material_check.md`（11 个资产 hash 全匹配、归档内部
  56 行清单自校验通过、个人路径 0 命中）。

## 已知缺口

- `A-API-SNAPSHOT` 的上游 commit 仍未确认（`GAP-API-1`）；当前按固定依赖包提供。
- 参考 ELF 的原始生成信息未恢复（`GAP-DATA-1` 的剩余部分）。
