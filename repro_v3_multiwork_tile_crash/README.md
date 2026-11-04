# Repro: QSMLA v3 CSA kernel crashes on any multi-work case (`kWorkCount > 1`)

## 1. 现象

`v3` CSA kernel（`quant_sparse_flash_mla_v3.hpp`，`IMPL=csa_v3_4pe`）在 **work 循环真正成环**
（`kWorkCount = B*S1*N2*GSliceCount > 1`）的任何 case 上，**gfrun 与 gfsim 都在模型里崩溃**：

```
FuncTBackToFrontIndex: out of index range
  func BackToFrontIndex, file common/DataStruct/BaseROB.h:101
```

- 此前所有 v3 case 都是 `B=1, S1=1, N1=64 → kWorkCount=1`（work 循环被完全展开），**从未触发**；
- 一旦 `kWorkCount>1`（任何 `B>1` / `S1>1` 的 case，例如生产网络 case `B=8,S1=4,N1=128 → 64 works`）必现；
- **与 N1 / G-slice / PA_BBND / TND 无关**：本复现用的是**未修改的 v3** + `N1=64`，仅 `B=64`。

## 2. 复现步骤

需要 `SuperScalarModel`（含 `bin/gfrun`、`bin/gfsim`）。本目录自带可复现 ELF。

```bash
ELF=<本目录>/csa_small_b64_v3.elf
cd <SuperScalarModel>

# 功能模型（gfrun）
./bin/gfrun -f "$ELF" -s softcore.multiThreadNum=4
# 期望（崩溃）：
#   gfrun: illegal instruction: ASSERTION FAILED: false
#   FuncTBackToFrontIndex: out of index range (BaseROB.h:101)

# 时序模型（gfsim）同样崩
./bin/gfsim -f "$ELF" -s softcore.multiThreadNum=4
```

- **ELF**：`csa_small_b64_v3.elf`（`csa_small_b64` case：`B=64, S1=1, S2=128, N1=64, N2=1, D=512, win127/0, cmp_s2=64, TopK=40, ratio=4, HIF8入/BF16出`，`Tk=128, Td=512`）。
- **对照**：同 kernel 的 `csa_small`（`B=1 → kWorkCount=1`）正常（gfrun 100% @2e-2，gfsim 26,022 cyc）。

## 3. 根因（已定位到编译器源码）

崩溃点在**模型 side**，但**触发代码由编译器发出**：

1. work 循环体顶部，编译器发出把“循环携带 tile”溢出到栈的指令
   `TSTORE <Col=16, S64>, T#2, [base=...]`。
2. 但在**最终发射流**里，这条 spill **之前只定义了 1 个 T-tile**（第 2 个 tile 的定义排在它之后）：

   ```
   814bcc: TEXPANDS <Row=16,Col=1,FP32,CUBE_M16>, a0, ->T<128B>   # 定义 #1
   814bf8: TSTORE <Col=16, S64>, T#1, [base=...]                   # spill（ok）
   814c12: TSTORE <Col=16, S64>, T#2, [base=...]                   # spill ← 模型在此崩
   814c26: TEXPANDS <Row=16,Col=1,FP32,CUBE_M16>, a0, ->T<128B>   # 定义 #2（在 spill 之后）
   ```

3. 模型的 per-type tile 重命名窗口（ROB）按“相对索引”解析：`BackToFrontIndex(1)` 在 `size=1` 时越界 → 断言。

**精确机制**（编译器源码）：`llvm/lib/Target/LinxV5/LinxV5TRegToOffset.cpp::rewriteMBB`

```cpp
// tile 的 IMPLICIT_DEF（循环携带 live-in，前导块把它标为未定义）
if (MI.isImplicitDef()) { ... TRegIdx[Reg] = -1; continue; }        // :1610-1615
// 之后对该 reg 的 use：
int64_t Index = CurIdx - TRegIdx[Reg];        // = CurIdx - (-1) = CurIdx + 1  :1632
Register NewReg = OffsetBase + (Index - 1);   // = Tile_TOS1 + CurIdx           :1637
```

即：编译器对“循环携带 tile”在前导块发 `IMPLICIT_DEF`（未定义），其 use 被编码为 `T#(CurIdx+1)`
—— 一个**运行时从未物化**的窗口槽。硬件允许读未定义寄存器（读到死值），但模型不允许。

MIR 佐证（`-mllvm -print-after=linxv5-treg-to-offset`）：

```
bb.1:  renamable $tile_t2 = IMPLICIT_DEF        <- 前导块：循环携带 tile 标未定义
bb.2.for.body:  liveins: ... $tile_t2           <- 作为循环 live-in 传入
...
PseudoTSTORE <S64>, killed $tile_tos1   -> T#1
PseudoTSTORE <S64>, killed $tile_tos2   -> T#2  <- 引用未物化槽
```

## 4. 建议修法

- **编译器（推荐）**：让“循环携带 tile 的 `IMPLICIT_DEF` use”绑定到一个**已物化**的窗口槽，
  或把该 `IMPLICIT_DEF` 物化为真实（零）tile 定义。
  本目录附 `compiler_fault1_fix.patch`（在 `rewriteMBB` 里把这类 use 的 `Index` 绑到窗口头），
  应用后**该断言消失**（已验证：增量重建工具链 + 重跑本 ELF）。
- **模型**：正确表示“循环携带、首迭代未定义”的 tile 槽（返回与消费操作数一致的合法描述符）。

> 注：`compiler_fault1_fix.patch` 只解决上面这一层；修掉后本 ELF 会前进到下一处
> （编译器“窗口旋转 TMOV”复制的槽描述符与模型记账不自洽 → `ValidateLocalTlsu`/TFMA），
> 那是另一层问题。

## 5. 环境

- 编译器：`linx-toolchain-build`（`LinxISA/llvm-project`, `dev-llvm_56`），`output/linx_blockisa_llvm_musl`
- 模型：`SuperScalarModel`（`bin/gfrun`, `bin/gfsim`）
- 内核：**未修改的** `quant_sparse_flash_mla_v3.hpp`（`IMPL=csa_v3_4pe`）

### 重新构建该 ELF（可选）

```bash
export COMPILER_DIR=<linx-toolchain>/output/linx_blockisa_llvm_musl/bin
cd benchmark/one-level-arch/test/solution/quant_sparse_flash_mla
make TESTCASE=quant_sparse_flash_mla QSMLA_SPARSE_EMBED_INPUT=on \
  B=64 s1=1 s2=128 N1=64 N2=1 D=512 wleft=127 wright=0 softmax_scale=0.04419417 \
  cmp_s2=64 ori_topk=40 cmp_topk=40 cmp_ratio=4 Tm=64 Tk=128 Td_block=512 \
  QSMLA_DTYPE=HIF8 QSMLA_SPARSE_DATA_ROOT=<data-root> \
  QSMLA_SPARSE_CASE=csa_small_b64 IMPL=csa_v3_4pe
```

`csa_small_b64` 数据由 `run_qsmla_cpu_reference.py --case csa_small_b64 --dtype HIF8` 生成。

## 6. 相关编译器文件

`LinxV5TRegToOffset.cpp`、`LinxV5TileFixup.cpp`、`LinxV5RegisterInfo.cpp`（`PseudoTSpill` 降低）、
`LinxV5InstrInfo.cpp:706/759`（溢出生成）。
