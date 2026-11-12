# 外部资产清单（NDF §3.7.4）

状态：`uploaded` = 已在本归档（git）内；`prepared` = 已生成并核对，但只在作者机器
（尚未发布）。`download_verified` 均未做。

| asset_id | 用途（required/optional） | 内容 | 字节 | SHA256 | 位置 | 状态 | 责任/核对日期 |
|---|---|---|---:|---|---|---|---|
| `A-COMPILER-PATCH` | required_for: build_reproduced | 编译器本地 patch（6 文件） | 46,475 | `5fef1a21a823c37e251b8fcc904837da25efd9546e9b6f4c3b9d691210553a5d` | `dependencies/compiler_llvm_pr146/compiler.patch` | uploaded（git） | lvhao / 2026-11-12 |
| `A-API-SNAPSHOT` | required_for: build_reproduced | PTO API 快照 169 文件 + 清单（发布包 `api_snapshot_r3.tar.gz`，213,721 B，sha256 `b3c6e97d…`） | 1863758（169 文件合计） | `256f5eb516f3a700e57f25dbf9fbe5463a437f6131902822026c873b45eef4d8`（MANIFEST 自身） | `outputs/hash256_soa_20261105/optimization/dependencies/tileop-api` | prepared（GAP-API-1） | lvhao / 2026-11-12 |
| `A-REF-ELF` | required_for: dataset_generation, replay | 参考 fixture ELF `hash_6144_PE4.elf` | 40,886,656 | `960ba161b8c662a74bf3f93ed5ad96a75f3375d4a91eee4607bc870b9fa48bd4` | `eval/perf_data/hashtable_lookup/` | prepared（GAP-DATA-1） | lvhao / 2026-11-12 |
| `A-R1-ELF` | required_for: replay_existing_elf（r1） | 64-lane 2+2 perf ELF | 61,306,432 | `2324ea4c438bb2eb3b6214190611c2c643b06338c348784a59d0301ef2e3b7d6` | `outputs/.../slots_sweep/elfs/adaptive_2p2_perf.elf` | prepared（GAP-EXT-1） | lvhao / 2026-11-12 |
| `A-R2-ELF` | required_for: replay_existing_elf（r2） | 256-lane 2+6 soa_wide ELF | 61,306,304 | `35218ed00c5f34cd2a30d0c8ef7fea66ecfd5047611575d397246813383f1e2f` | `A-R2-ELF`（prepared；可由 build 脚本重建） | prepared（GAP-EXT-1） | lvhao / 2026-11-12 |
| `A-R3-ELF` | optional_for: compile_evidence | operator adaptive AoS perf ELF | 40,860,136 | `38a5f34af9b0ab65d690d4c2766319aef5bde991f72933a4a5fa31933214476b` | `A-R3-ELF`（prepared；可重建） | prepared（GAP-EXT-1） | lvhao / 2026-11-12 |
| `A-R1-TRACE` | required_for: perf_trace（r1/r2 代表运行；rw8/way7 未采） | r1 full SwimLane（502 MB）+ nocounters + swim.log，gzip | 16,838,669 / 6,601,655 / 202,860 | `d46ff56a1613fc72d3f4e8a56a58a450db7d058c7dfc8a9733ee53ab9ba94fdc` / `ba3a268c30f9ef130688fd4242c0a0468ced2a1ef49df350e0d656c1a8c072da` / `5276092461c4e6672da26848b3068cdddb0e1d89c77604edf234a92b0fbc4da6` | `outputs/hashtable_lookup_archive_assets/r1_rand300_01/` | prepared（GAP-EXT-1） | lvhao / 2026-11-12 |
| `A-R2-TRACE` | required_for: perf_trace（r1/r2 代表运行；rw8/way7 未采） | r2 full SwimLane（291 MB）+ nocounters + swim.log，gzip | 9,291,447 / 3,332,014 / 200,869 | `d5a867bce24fe4957670d593e9c07851ca3c58ccdd8ec0733eccbf5b4e2a94ed` / `66fc11410aa844b6449d4312d04b24353c1c4e9aae9a2e9444108b3f53e8fcc5` / `4b9e7b5b2d8a572b3b5226a69d2dc64de0ab09bb29c6026c4e11fdb42e49f607` | `outputs/hashtable_lookup_archive_assets/r2_rand300_rw48/` | prepared（GAP-EXT-1） | lvhao / 2026-11-12 |

## 组合与使用

- 重放 r1：`A-R1-ELF` + 模型 runtime（`dependencies/simulator_pr1177`）+ `config/random300.json`
  + 输入（`dependencies/dataset`）；trace 仅用于审阅。
- 重放 r2：`A-R2-ELF` + 同一模型 + `config/random300.json` + `sl2.RWDB_ENTRY_COUNT=48`。
- 重建：`A-COMPILER-PATCH` + 上游 `llvm-project@a75987bcc` + `A-API-SNAPSHOT` + 数据/参考 ELF。

## 获取方式与缺口

- 目前只有 `A-COMPILER-PATCH` 在 git 内；其余资产均为 `prepared`（作者本机路径），
  **尚未上传 Release**（R017/R018 的部分满足）。获取方式、访问条件与 `download_verified`
  需在发布后补。

## revision 3 发布包（prepared，URL 待 Release 后补）

| asset_id | 内容 | 字节 | SHA256 |
|---|---|---:|---|
| `A-R1-TRACE` | `r1_trace_r3.tar.gz`：sanitized log + meta + full/nocounters gz | 16,155,560 | `c9593e7eb251b4e323d45a104468c73c1fc7e2985b6006564d79a6f1e765f553` |
| `A-R2-TRACE` | `r2_trace_r3.tar.gz`：同上（rw48） | 8,156,680 | `ae076cb478a78e1c113f9a434b6f294c432c3feacc5d58e103801bf47b4c0d6f` |
| `A-R2-WAY7-TRACE` | `r2_way7_trace_r3.tar.gz`：way7 完整 trace 集 | 7,928,613 | `9b4deb3baa62a78af1c0baa3c2e1ed1fb4b9d91b20bd098e8bf4d87c2b3d749a` |
| `A-R2-RW8-TRACE` | `r2_rw8_trace_r3.tar.gz`：rw8 完整 trace 集 | 8,491,946 | `c2d870619ea8ae4e18da29216f7ab6e2ed8577df26c3a800afe2f8091356059f` |
| `A-PIPEV` | `pipeview_r3.tar.gz`：r1/r2 `.out.gz` + 日志 + `/usr/bin/time -l` | 22,624,341 | `6738fefe9d6540aed43db1e6feb8e470670dde9f7e22a2c18fcf0c5bf4ef04b7` |
| `A-API-SNAPSHOT` | `api_snapshot_r3.tar.gz`：169 文件快照 | 213,721 | `b3c6e97dd3a5d1650bbe2d5848c60873868cf977abba2667a78080d4968032d6` |
| `A-ARCHIVE-R3` | `hashlk_archive_r3.tar.gz`：本归档目录（含清单） | 待生成 | 待生成 |

脱敏规则与原件/派生 hash 见 `outputs/hashtable_lookup_archive_assets/release_r3/sanitization.json`
和 `validation/PATH_SCAN.md`；trace JSON 内容未改。
