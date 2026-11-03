# Hash dataset performance / verification

固定数据：6144查询、2550000槽、70%装载率。保持原始hash算子，无优化叠加。

## 独立模式

```sh
python3 /Users/blacktraker/Programming/gitproj/DV4/SuperNPUBench/benchmark/one-level-arch/test/kernel/hashtable_lookup/run_dataset.py --mode perf
python3 /Users/blacktraker/Programming/gitproj/DV4/SuperNPUBench/benchmark/one-level-arch/test/kernel/hashtable_lookup/run_dataset.py --mode verify
```

- `perf`：编译独立ELF，不包含golden、数值比较、哨兵初始化或校验同步。每个PE完成算子后写finisher，模型等待4个PE完成。finisher的0x5555仅表示完成；JSON中的`numerical_verified`为null，不报告数值PASS。
- `verify`：独立ELF，PE0初始化边界哨兵，4 PE完成后逐项比较6144个结果与lookup_golden.bin，检查两侧哨兵，所有PE写出共享错误状态。只有零错误才报告`numerical_verified: true`。
- `--completion-barrier`：实验选项，仅 perf 在算子返回后等待全部 PE 再写完成标志；不包含数值校验，统计窗口仍截止最后结果 TSTORE。默认关闭，A/B 需一致。
- `--kernel-header /absolute/path/kernel.hpp`：选择实验算子头文件，perf/verify 应使用同一个头文件；默认仍为原始算子。
- `--build-only`：仅编译和生成反汇编；`--output-dir /absolute/path`选择输出根目录，每个模式使用独立子目录。
- 两种模式均使用freestanding启动，不打开`res_check=on`，不链接libc。`verify`使用显式校验。
- `perf`自动生成swimlane.json.gz。算子cycle=最后结果TSTORE完成-首次输入TLOAD开始。finisher、返回和退出不计入算子耗时。Total Cycles仅作程序级参考。

Make入口也支持`hash_mode=perf`和`hash_mode=verify`，默认`hash_mode=smoke`保留原139-key测试。切换模式会强制重新编译dataset对象，避免同OBJ_ROOT复用错误模式。编译必须设置COMPILER_DIR。

## Tile粒度说明

32 key/PE是当前实现与工具链profile的选择，不是MurmurHash或线性探测算法要求。当前U32状态是32×1=128B，U64 key是32×1=256B；不能把所有tileop都称为128B。

|每PE处理key数|M32逻辑形状|U32数据量|U64 key数据量|
|---|---|---|---|
|32（当前）|32×1|128B|256B|
|64（候选）|32×2|256B|512B|
|128（候选）|32×4|512B|1024B|

这里的数据量是逻辑载荷，不能等同于模型内部每个uop的执行粒度。较大的tile指令仍可被拆成多个CELL/uop执行。

本地编译器API的ElementTile允许U32/S32/F32 logical32/128，但U64仅logical32。编译器element访存检查也明确要求i64访问使用logical32；所以此hash的U64 key gather路径不能只改模板为64/128就获得大tile支持。

本地ISA的M32 GPR谓词每个word覆盖32行×2列。64 key的32×2形状处于该几何范围，但并不证明当前编译器已支持完整U64 element lowering。128 key的32×4需要相应的PredicateTile或分片谓词方案，不能原样沿用当前一个GPR判断全部active的写法。ExecutionMask的谓词Tile形式支持更大的匹配形状；ISA不存在一概限制所有tileop为128B的规则。

增大tile能摊薄block/标量控制开销并提供更多CELL/uop，但不会消除随机gather等待。当前批内有效probe lane比例16.83%，放大批次可能加重最长链拖尾，需要实测总cycle与有效lane比例，不能仅看VEC长条是否更满。

证据：

- /Users/blacktraker/Programming/gitproj/DV4/linx-toolchain-build/output/linx_blockisa_llvm_musl/lib/clang/15.0.4/include/tileop-api/common/pto_tile.hpp:1292
- /Users/blacktraker/Programming/gitproj/DV4/linx-toolchain-build/src/llvm-project/llvm/lib/Target/LinxV5/LinxV5ElementRegion.cpp:2399
- /Users/blacktraker/Programming/gitproj/DV4/pto-spec/asl/tile/model/legality/predicate-carriers.asl:36
- /Users/blacktraker/Programming/gitproj/DV4/pto-spec/docs/block/model/dispatch/execution-mask-schema.md

## 2026-11-02 验证结果

- perf：4 PE完成，输入TLOAD=192，结果TSTORE=192；首TLOAD=338，末TSTORE完成=291870，算子291532 cycles，程序总计291896。numerical_verified=null。
- verify：6144输出和两侧哨兵比较通过，4 PE均0x5555，numerical_verified=true，程序总计421564。
- 新perf入口与之前带校验入口的二进制、启动调度不同；291532与旧295231不可直接用来声称算法优化。后续优化应统一使用独立perf入口作A/B。
- [/Users/blacktraker/Programming/gitproj/DV4/SuperNPUBench/outputs/hash_modes_20261102/perf/result.json](/Users/blacktraker/Programming/gitproj/DV4/SuperNPUBench/outputs/hash_modes_20261102/perf/result.json)
- [/Users/blacktraker/Programming/gitproj/DV4/SuperNPUBench/outputs/hash_modes_20261102/verify/result.json](/Users/blacktraker/Programming/gitproj/DV4/SuperNPUBench/outputs/hash_modes_20261102/verify/result.json)
- [/Users/blacktraker/Programming/gitproj/DV4/SuperNPUBench/outputs/hash_modes_20261102/perf/swimlane.json.gz](/Users/blacktraker/Programming/gitproj/DV4/SuperNPUBench/outputs/hash_modes_20261102/perf/swimlane.json.gz)
