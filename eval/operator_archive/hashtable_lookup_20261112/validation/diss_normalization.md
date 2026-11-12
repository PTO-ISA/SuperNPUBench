# 反汇编文件头规范化（NDF §5.2）

`perf.diss` 由 `llvm-objdump -d <ELF>` 生成，第 2 行是输入文件路径。为发布材料去掉
临时路径，只把该行替换为 `perf.elf:`，指令正文未改。

| 文件 | 原 sha256 | 新 sha256 | 变更 |
|---|---|---|---|
| `results/r2_256l_2p6_rwdb48/perf.diss` | `7213026eb93d98c5b3174172fb123f2da605a4a1c5b2048375aea3ad37c6d8ec` | `ab7b75629305549af796070442b326b02bb7fc8e460395264294499c0e7354ef` | 第 2 行：`/tmp/soa_op/perf.elf:` → `perf.elf:` |
| `results/r3_operator_adaptive_aos/perf.diss` | `a320efe59820b54883ebd2da4b58558ca4365662c2ace2131d48cb3c785f34f3` | `9137adb640df81aecd2613cd2b2989ac0fb736880206a0dd1ce4743361622a24` | 第 2 行：`/tmp/hash_op26/..._perf_PE4.elf:` → `perf.elf:` |

核验：对 `origin/main` 的同名文件做同样替换后，与本归档当前文件逐字节一致；
`diff <(tail -n +3 原文件) <(tail -n +3 新文件)` 为空（指令正文一致）。ELF 未重新生成，
哈希见 `results/*/manifest.json`。
