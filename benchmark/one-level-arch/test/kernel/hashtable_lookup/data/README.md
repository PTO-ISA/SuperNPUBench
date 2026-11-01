# Hash lookup 数据

生成脚本：/Users/blacktraker/Programming/gitproj/DV4/SuperNPUBench/benchmark/one-level-arch/test/kernel/hashtable_lookup/generate_data.py

按用户脚本：2,550,000槽、1,785,000次插入、6144个查询，查询从插入key无放回采样，因此全部命中。表项为小端 `<Qi4x>`，空key为UINT64_MAX，空value和padding为0。

- table_data_70pct.bin：40,800,000字节表。
- input_keys_data.bin：6144个小端U64查询。
- golden_data.bin：原脚本语义，MurmurHash3(key)%2550000的初始槽位，不是lookup结果。
- lookup_golden.bin：6144个小端S32查找结果，已逐项与原始key/value校验。
- lookup_probe_counts.bin：每个query的读取次数，包含首次读取。
- const_mem.bin：保留用户原始18个U32参数，固定地址0x10000000/0x20000000/0x40000000；仅生成文件，不自动映射到当前ELF或hosted ABI。
- init_keys.bin / init_vals.bin：保存随机初始输入。
- manifest.json：尺寸、SHA256和探测统计。

seed=0按原脚本不设置随机种子。重现当前数据请使用保存的初始输入：

```sh
python3 /Users/blacktraker/Programming/gitproj/DV4/SuperNPUBench/benchmark/one-level-arch/test/kernel/hashtable_lookup/generate_data.py --reuse-init --output-dir /Users/blacktraker/Programming/gitproj/DV4/SuperNPUBench/benchmark/one-level-arch/test/kernel/hashtable_lookup/data
```

2026-10-31 已用当时保存的数据在本地修复后的 gfsim 通过6144项数值校验。生成的二进制与manifest不纳入Git；首次运行生成脚本不传 --reuse-init。seed=0不固定随机输入，重新生成的数据不保证与该次验证一致。
