# ISSUE: emulator NORM Local TSTORE 不支持打包 4-bit —— 描述符 size 与源行 stride 均按字节容器口径（应按元素位宽），与 pto-spec `DerivedTileRows` / `TileMemoryElementAddress` 不一致

- **归属仓库**: SuperScalarModel（gfrun / emulator）【SuperScalarModel issue557】
- **状态**: 已定位根因（对照 pto-spec 权威 ASL），未修
- **对应 RECORD**: 问题26

## 组件清单（暴露此问题的版本组合）

| 组件 | 版本 | 备注 |
|---|---|---|
| SuperScalarModel（gfrun） | 分支 `codex/consolidate-post-main-fixes-20260903`，tip `49547742` | 缺打包 4-bit NORM TSTORE 支持 |
| 工具链 llvm-project | `dev-llvm15_56` @ `1ae4ee39` | clang 15.0.4 |
| 工具链 Linx-TileOP-API | `linx` @ `804eb03` | 含官方 `ddd07b9`（TCVT 目的 logical tile size），fp4 TCVT 形状自洽 |
| 工具链 musl | `af0dfc20` | — |
| Bench 算子代码 | PR: https://github.com/PTO-ISA/SuperNPUBench/pull/111 （fork 分支 `ziyang-cheng/SuperNPUBench:dmxq-ops-20260904`） | 触发 kernel/驱动/金标脚本 |
| pto-spec（规范核对） | `6e3e056f` | `DerivedTileRows` / `TileMemoryElementAddress` |

## 前置依赖（到达本问题所需的其它 issue）

在干净 `49547742` 上，`tail_ocp_fp4` res_check 会**先**撞两道更前置的墙，须先解开才能到达本问题：
1. `ppoll` handler 缺失（**SuperScalarModel issue554**）—— 否则 hosted musl 启动即 abort。
2. compare/select 源按 dtype 相等误杀 reinterpret 视图（**SuperScalarModel issue254** 的 compare/select sibling，`IsCompatibleDataTile`）—— 否则 fp4 三守卫的 `TCMPS` 先崩。

解开上述两处后，执行流到达 data pass 末尾的打包 fp4 `TSTORE`，即本问题。

## 一、现象

`dynamic_mx_quant_tail_ocp_fp4`（bf16 in / 打包 fp4 out / e8m0 scale，4-PE，512×256）data pass 末 `TSTORE(gy, oq)`（`oq` = fp4，RowMajor/NORM）分两层暴露：

**(a) 描述符崩**（`emulator/engine/AccumulateBlockInfo.cpp` `IsLegalLocalTileDescriptor`）：
```
gfrun: ASSERTION FAILED: ... IsLegalLocalTileDescriptor(inst->srcs[1]) &&
       "Local TSTORE requires one legal source Tile descriptor"
```
诊断出 fp4 源 tile：`dtype=FP4 row=128 col=32 validRow=64 validCol=32 size=2048`。`IsLegalLocalTileDescriptor` 末行按字节口径校验 `size == rows*cols*BytesOf`（`BytesOf(FP4)=1`）→ 期望 `128*32*1=4096 ≠ 2048` → 拒。
> 注：`row=128` 是 emulator 按元素位宽（4-bit）从 2048B 容量派生（见 §二）；`validRow=64` 是 kernel 实际行数。

**(b) 描述符放行后行错位**（`emulator/engine/TMAEngine.cpp` `ExecuteTSTORE` NORM 分支）：output=fail，仅 **tile 第 0 行落对、其余行全错位**：
```
dynamic_mx_quant_tail_ocp_fp4: output=fail (MSE=8.389874, MaxAE=12.000000), scale=pass
```
逐行核对：512 行里只有 `[0,64,128,192,256,320,384,448]`（各 PE 各 64 行 tile 的第 0 行）逐字节正确，其余行全错。

## 二、复现步骤

```bash
export COMPILER_DIR=/path/to/linx-toolchain-build/output/linx_blockisa_llvm_musl/bin
export LINX_SYSROOT=/path/to/linx-toolchain-build/output/linx_blockisa_llvm_musl/sysroot/usr
cd SuperNPUBench/benchmark/one-level-arch/test/kernel/multi_thread/quant/dynamic_mx_quant

# 1) 金标（bf16 in / OCP tail / FP4 / compact scale）
python3 src/gen_dynamic_mx_quant_data.py --M 512 --K 256 --block-size 32 \
    --algo OCP --kernel tail --dtype FP4 --in-dtype bf16 --scale-layout compact --seed 42 \
    -o ../../../../compare/dynamic_mx_quant_tail_ocp_fp4

# 2) 编 res_check ELF
make TESTCASE=dynamic_mx_quant TYPE=TAIL_OCP_FP4 res_check=on diss

# 3) 4 线程 gfrun（前置两 issue 解开后）→ (a) 崩描述符；描述符放行后 → (b) 落盘行错位
bin/gfrun -f .../dynamic_mx_quant_tail_ocp_fp4.elf -s softcore.multiThreadNum=4 -t 1

# 4) 比对（(b) 阶段）
python3 src/dynamic_mx_quant_data_compare.py -d .../dynamic_mx_quant_tail_ocp_fp4.elf \
    --dtype FP4 --scale-layout compact --cmp-root ../../../../compare
```

## 三、根因（对照 pto-spec 权威 ASL）

pto-spec 对打包 4-bit 一律用**元素位宽**、按 `element DIV 2` 打包寻址，而非字节容器口径：

- **描述符 size / 行数**：`DerivedTileRows`（`asl/tile/model/shape/rows-columns.asl`）：
  ```
  row_bits = columns * TileElementBits(data_type)     // TileElementBits(E2M1X2)=4
  rows     = (capacity_bytes * 8) DIVRM row_bits
  ```
  即容量与行数按 **`TileElementBits`（4-bit）** 关联。故描述符 size 校验须 `size == (rows*cols*TileElementBits + 7)/8`，而非 `rows*cols*BytesOf`（`BytesOf(FP4)=1` 会期望 2× 存储）。

- **内存元素地址 / 行 stride**：`TileMemoryElementAddress`（`asl/tile/model/memory/addressing.asl`）：
  ```
  if FourBit: offset = element DIVRM 2        // 2 元素 / 字节
  TileMemoryElementHighNibble = (element MOD 2 == 1)
  ```
  即线性元素 `element = row*cols + col` 的字节地址 = `(row*cols + col) DIV 2`；行 stride = `cols/2` 打包字节。而 NORM `ExecuteTSTORE` 用 `srcRowWidth = totalCol * eleSize`（`eleSize=1` 字节容器）= `totalCol`，是打包行宽的 **2×** → 第 i 源行读到 `i*totalCol`（应 `i*totalCol/2`），仅 i=0 对。

**对比**：CUBE TSTORE 分支已有 `packed = CubeCellElementBits(dt)==4; memoryRowBytes = packed ? (validCol+1)/2 : ...` 打包处理；**NORM 分支缺**同等处理。列内 nibble 打包（`EleOffset`=idx/2 + `EleDataExtract` nibble mask）本就正确，唯**描述符 size 校验**与**源行 stride**按字节口径。

## 四、修复建议

1. `IsLegalLocalTileDescriptor`（NORM/RowMajor 分支）：size 校验改按元素位宽 `size == (rows*cols*ElementBitsOf(dt) + 7)/8`。字节类型 `ElementBitsOf==BytesOf*8`，与原判定等价，仅打包 4-bit 生效。
2. `ExecuteTSTORE`（NORM 分支）：`srcRowWidth` 与无 B.IOR 时的 GM 回退 stride 改按打包元素地址（`floor(totalCol * EleRealSize)`，等价 `TileMemoryElementAddress(0, totalCol, dt)`）。镜像 CUBE 分支的 `packed ? (validCol+1)/2` 处理。

## 五、影响范围

所有把打包 4-bit（FP4/FP4_1/HIF4/INT4/UINT4）经 **NORM Local TSTORE** 落盘的路径（`dynamic_mx_quant` 的 tail/nontail OCP-FP4 全部）。scale pass（e8m0）与 CUBE 布局 TSTORE 不受影响。
