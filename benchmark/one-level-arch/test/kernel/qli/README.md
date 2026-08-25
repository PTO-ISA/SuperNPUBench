# QLI radix-select TopK — 验证流程

> QLI 量化闪电索引（Quant Lightning Indexer）TopK demo 的构建与精度验证流程。
> Kernel：`kernels/qli/qli_pto_opt.hpp`（`qli_topk_radix`，多轮 MSD radix-select）。
> 精度判据：**TopK set match**（无序集合一致）+ score cosine。

## Environment requirements（SuperScalarModel 本地修复）

本 demo 的多轮 `THISTOGRAM`（Byte2/1/0 前缀收窄）、UINT32 域
`TROWSUM/TROWMAX/TROWARGMAX`、`TROWARGMAX` 提取依赖 SuperScalarModel 侧的
**本地修复**（已在本地验证环境应用，见
[`kernels/qli/qli_radix_issues_found.md`](../../../kernels/qli/qli_radix_issues_found.md)）：
在 stock 仿真器上需先应用以下改动才能全功能运行：

| 项 | 文件 | 内容 | 对应 issue |
|---|---|---|---|
| THISTOGRAM ByteId 解码 | `isa/Block.cpp` `HandleBDATR` | 从 bits[19:18] 读 selectedByte（而非 bits[28:27]），否则 Byte0/1/2 恒解码为 Byte3 | R1 |
| UINT32 归约/argmax 门禁 | `emulator/engine/AccumulateBlockInfo.cpp` `IsReduceAndExpandTeplDataType` | 允许 UINT32/UINT16 用于 `TROWSUM/TROWMAX/TROWARGMAX`（B8） | R2/B8 |
| TROWARGMAX 指令映射 | `isa/ISACommon/TileOpManager.h` `GetTeplTileOp` | `REDUCEANDBROADCAST_RESERVE_{1..4}` → TROWARGMAX/TROWARGMIN/TCOLARGMAX/TCOLARGMIN | R3（A4） |

> 未应用上述修复时，ByteId<3 轮次会统计错误字节、UINT32 掩码归约会被门禁拒绝，
> 导致 TopK 结果错误或中止——**请先按 issues_found.md 应用修复再运行本 demo**。

## 预览

本目录是 `test/kernel/qli`，对应源文件在 `src/`：

| 文件 | 作用 |
|---|---|
| `src/qli_check_opt.cpp` | 驱动：Step1-6（ql_pto）+ Step7（ql_topk_radix），含 Step7 独立 TRACE marker |
| `src/gen_qli_golden.py` | 生成输入 .bin + numpy golden；`--mode verify` 支持 set-match 判据 |
| `src/qli_check_data.s` | 用 `.incbin` 嵌入 5 个输入 .bin（Q/K/W/scale_q/scale_k） |
| `src/fix_cpp_addrs.py` | 从 ELF 符号更新 `qli_check_opt.cpp` 的 SRC*_ADDR 绝对地址 |
| `Makefile` | TESTCASE 构建入口（`qli_check_opt` 为 demo 主线） |
| `compile.all` | 已验证的关键 Case 配置 |

## 关键 Case（已验证 set-match 100%）

| Sq | Skv | topK | chunks | 说明 |
|---|---|---|---|---|
| 64 | 128 | 128 | 1 | 单 chunk，多 token，全量 topK |
| 4 | 2048 | 512 | 1 | 单 chunk，大 topK |
| 4 | 8192 | 512 | 4 | 4 chunk 大规模 |
| 4 | 2080 | 512 | 1+tail | tail chunk（Skv%2048=32，Skv%kTk=0） |

> 注：Step1-6 的 `Skv%kTk==0` 是硬约束；`Skv=2056`（非 kTk 倍数）时 ql_pto
> 仅计算 2048 列，后续列 score 为未定义，不能用作 topK=512 验证。因此 tail
> 验证选用 `Skv=2080`（2048+32，2080%32=0）。

## 构建与验证步骤

> 需要先设置 `COMPILER_DIR`（linx_blockisa_llvm_musl 工具链 bin 目录）。

### 0. 定义本用例参数变量（后续命令引用）

```bash
# 与 Makefile 维度一致
export SQ=4 SKV=8192 TOPK=512 G=64 TM=16 TK=32
OUTDIR=../../../../../compare/qli_fp8_B1_Sq${SQ}_Skv${SKV}_g${G}_Tm${TM}_Tk${TK}
```

### 1. 生成输入与 golden（维度以 --sq/--skv/--topk 显式传入）

```bash
cd test/kernel/qli/src
python3 gen_qli_golden.py --mode gen \
    --sq $SQ --skv $SKV --topk $TOPK --g $G --d 128 \
    --outdir $OUTDIR
```

> 维度参数即真实张量形状；`--outdir` 仅决定存放目录，不影响形状。

脚本会写入 `compare/<cfg>/` 下：
- `srcq.bin` / `srck.bin` / `srcw.bin` / `srcsq.bin` / `srcsk.bin`（输入）
- `reference_scores.bin` / `reference_indices.bin`（numpy golden）

### 2. 更新数据嵌入文件路径

`qli_check_data.s` 中的 `.incbin` 指向 `compare/<cfg>`。默认提交指向
`qli_fp8_B1_Sq64_Skv128_g64_Tm16_Tk32`；验证其他配置时替换**整条路径前缀**：

```bash
cd test/kernel/qli/src
# 生成 .incbin 路径（repo 相对 .s；此处直接以绝对路径为例，也可替换为自己 checkout 的 repo 根）
REPO_ROOT=$(git rev-parse --show-toplevel 2>/dev/null || echo /path/your/checkout)
# 确保 .incbin 使用当前机器上真实的 compare 路径：
sed -i "s|/home/z00947698/SuperNPUBench|$REPO_ROOT|g; \
        s|qi_fp8_B1_Sq[0-9]*_Skv[0-9]*_g64_Tm16_Tk[0-9]*|qli_fp8_B1_Sq${SQ}_Skv${SKV}_g64_Tm16_Tk${TK}|g" \
    qli_check_data.s
```

> 注意：`qli_check_data.s` 存库时使用示例绝对路径；任何 checkout 上必须先执行
> 本 sed 将路径替换为你的 `compare` 目录，再汇编。

### 3. 汇编数据对象

```bash
cd test/kernel/qli/src
$COMPILER_DIR/clang -c -x assembler qli_check_data.s -o ../src/qli_check_data.o
```

### 4. 构建 ELF

```bash
cd test/kernel/qli
make TESTCASE=qli_check_opt QLI_DTYPE=FP8 Sq=4 Skv=8192 topk=512 Tm=16 Tk=32
```

### 5. 修复 .data 段符号地址（绝对地址，链接后漂移需迭代）

`qli_check_opt.cpp` 使用 `.data` 段绝对地址（P1 优化，无 copy_bytes）。
链接后地址会漂移，用 `fix_cpp_addrs.py` 回写并重新编译，迭代至稳定：

```bash
export NM=$COMPILER_DIR/llvm-nm
ELF=../../output/kernel/qli/elf/kernel_qli/qli_check_opt_fp8_B1_Sq4_Skv8192_g64_Tm16_Tk32.elf
# 循环：make → python3 src/fix_cpp_addrs.py src/qli_check_opt.cpp $ELF $NM → make
# 直到两次 llvm-nm 的 srcq 地址一致（通常 1-2 轮收敛）
```

### 6. gfrun 运行 + 内存 dump

```bash
# OUT_SCORES=0x4000802000, OUT_INDICES=OUT_SCORES+Sq*Skv*4（驱动内已动态化）
DUMP_SIZE=$(( SQ * SKV * 4 + SQ * TOPK * 4 + 65536 ))
$SSM/bin/gfrun -f $ELF \
    --dump-memory 0x4000802000:${DUMP_SIZE}:${OUTDIR}/sim_out_radix.bin
```

> `$SQ/$SKV/$TOPK/$OUTDIR` 需在步骤 0 中定义（见前文）；否则 dump 尺寸会算错。

### 7. 精度验证（set-match 主判据）

```bash
python3 - <<'EOF'
import numpy as np
SQ, SKV, TOPK = 4, 8192, 512
data = open('sim_out_radix.bin', 'rb').read()
scores = np.frombuffer(data[:SQ*SKV*4], dtype=np.float32).reshape(SQ, SKV)
indices = np.frombuffer(data[SQ*SKV*4:SQ*SKV*4+SQ*TOPK*4],
                        dtype=np.int32).reshape(SQ, TOPK)
ref = np.fromfile('reference_scores.bin', dtype=np.float32).reshape(SQ, SKV)
ref_idx = np.fromfile('reference_indices.bin', dtype=np.int32).reshape(SQ, TOPK)
mask = scores > -1e29
c = float(np.dot(scores[mask], ref[mask]) /
          (np.linalg.norm(scores[mask]) * np.linalg.norm(ref[mask])))
setm = sum(1 for i in range(SQ) if set(indices[i].tolist()) == set(ref_idx[i].tolist()))
print(f'cosine={c:.6f} set={setm}/{SQ}')
EOF
```

> 通过条件：`cosine≈1.0` 且 `set == SQ`（set-match 100%）。

## Scratch memory contract（qli_topk_radix 内部工作区）

`ql_topk_radix` 在 `indices_gm` **之后**声明未隐式契约的工作区，调用方需保证
该区域可写且足够大：

| 区段 | 大小 | 偏移（相对 indices 起始） |
|---|---|---|
| indices 输出 | `Sq*topK*4` | 0 |
| key_scratch | `Sq*Skv*4` | `Sq*topK*4 + 8192` |
| temp_hist | `256*4` | `key_scratch + Sq*Skv*4` |
| prefix_buf | `32*4` | `temp_hist + 256*4` |

要求：`indices_gm` 之后连续可写空间 ≥ `Sq*topK*4 + 8192 + Sq*Skv*4 + 1152`
字节。demo 驱动（`qli_check_opt.cpp`）使用 map-memory 大区域满足该约束；
作为 header 集成时调用方按此分配（详见
[`qli_radix_issues_found.md` R 项 / kernel 源码注释）。

## 参考

- Kernel：`kernels/qli/qli_pto_opt.hpp`（`ql_topk_radix`）
- 设计：`kernels/qli/qli_pto_opt_histogram_radix_design.md §14`
- 已知问题：`kernels/qli/qli_radix_issues_found.md`
- 关键历史结果：`qli_fix_record.md §16`（superScalar 根目录）