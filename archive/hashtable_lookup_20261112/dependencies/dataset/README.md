# 依赖：数据集与参考 ELF

| 文件 | sha256 | 来源 |
|---|---|---|
| `keys32.bin`（2,550,000 × u32 + 128 guard） | `cefe04786c0ccee7b0091a62de78197a58d2b6f48d89617f23c04348a70e43f3` | `outputs/hash256_batchmap_20261105/data/` |
| `values.bin`（2,550,000 × i32 + 128 guard） | `511719ca60044a93c960f5b2789d487ffd18336ffeeb87711a3d567e61e4f448` | 同上 |
| `batch_map.bin`（24 × u32，256-lane） | `abea4b6fca0f936df9b564cea4e13c94b950eeeee4c37435b03e4148b5f18cec` | 同上 |
| `map64.bin`（96 × u32，64-lane） | `f59fa8c79f16491d6303bf72eb3bf8d8fb25ca8a09cb035a3207f25a526f06a6` | `.../slots_sweep/repro_matrix/maps/` |
| `map128.bin`（48 × u32，128-lane） | `2295894a335de5fafa34e5832f807b01fb385229be28518deb13d17f2c919df0` | 同上 |
| 参考 ELF `hash_6144_PE4.elf` | `960ba161b8c662a74bf3f93ed5ad96a75f3375d4a91eee4607bc870b9fa48bd4` | `eval/perf_data/hashtable_lookup/`（未上库） |

生成：`keys32/values` 由归档脚本从参考 ELF 的 AoS 表（偏移 0x1000）拆分（`gen_soa.py`，
见 `outputs/hash256_soa_20261105/`）；batch map 由 probe 长度离线均衡生成
（`tools/`）；参考 ELF 的原始生成信息见 `eval/perf_data/hashtable_lookup/README.md`。

## 缺口

- `GAP-DATA-1`：参考 ELF 未上库，keys/values 的生成链依赖它；仅凭本归档的内置哈希无法
  独立重建数据集。
