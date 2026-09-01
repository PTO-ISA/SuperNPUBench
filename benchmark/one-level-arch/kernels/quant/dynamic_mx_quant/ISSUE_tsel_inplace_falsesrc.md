# [Issue] 就地 `TSEL` 的 false-source（prior-dst）被读成 0 —— 崩溃修复之后仍存在的语义层 【已由 Linx-TileOP-API PR #41 根治】

## ✅ 解决（2026-09-01）：Linx-TileOP-API PR #41 `072ea70`

上游 **PR #41 `fix(tsel): emit canonical explicit false-source binding`** 从**工具链侧**根治：
把 `TSEL` 从「单拍 B.IOT + dst 就地 false-source」改成**两拍显式 false-source 绑定**——

```
B.IOT %[Mask], %[True], mask=1111                        // 第一拍：mask + true
B.IOT %[Prior], mask=1111, last, ->%[Dst]<%Z[TileSize]>  // 第二拍：Prior(false) + ->dst
    [Prior] "0"(dst.data())   // 用 "0" 匹配约束把 Prior 绑到 dst 的旧值（发布新 dst 前快照）
```

这样 emulator 的**原始**显式三源建模（`srcTile.size()==3`，false-source = `srcTile[2] = Prior = dst 旧值`）
即正确工作。**实测**：安装头应用 PR #41 后，`tsel_inplace_falsesrc_probe`（mask=0）吐 `0x1234`（保留 prior）、
`tsel_inplace_probe`（mask=1）吐 `0x1234`（选 true）、`dynamic_mx_quant_tail_ocp_fp4` **带三守卫** 4PE res_check
`output=pass (MaxAE=0.5) / scale=pass`。

**模型侧无需改动**：本地一度加的 `ab822e7a`（validate 放宽）+`1f398190`（execute 读 dstTile[0]）是针对**损坏
单拍**的 workaround，PR #41 后已**还原**——原始 d8903938 模型直接处理显式两拍形式。故本 issue 的修复**纯在
工具链**（应用 PR #41 到 installed 头）。

> 下方为定位过程原文，保留备查。

---

# [原始] 就地 `TSEL` 的 false-source（prior-dst）被读成 0 —— 崩溃修复之后仍存在的语义层

## 一句话

`TSEL(dst, mask, trueSrc)` 的就地语义是 `dst = mask ? trueSrc : dst_prior`（false-source 就是 dst
自身的先前值）。当 `mask` 全 0 时结果应 == dst 的先前值（一个 no-op）。但在当前 `SuperScalarModel`
（d8903938 + 本地 [[ISSUE_tsel_inplace_lowering]] 崩溃修复 `ab822e7a`+`1f398190`）上，就地 TSEL 的
**false-source 读到的是 0，而非 dst 的先前值** → 哪怕 `mask=0` 也把 dst 清零。

这是 [[ISSUE_tsel_inplace_lowering]]（问题18，单拍就地 B.IOT vs 模型显式三源 → **崩溃**）**更深一层**：
`ab822e7a`（validate 放宽）+ `1f398190`（execute 侧改读 `dstTile[0]`）**止住了崩溃**，但让模型去读
`dstTile[0]` 时，那里并不持有 dst 的先前值（读出 0），于是语义仍错。

## 最简复现探针

`test/kernel/quant/dynamic_mx_quant/src/tsel_inplace_falsesrc_probe.cpp`
（Makefile `TYPE=TSEL_INPLACE_FALSESRC_PROBE`）——整个 kernel 只有一条 TSEL：

```cpp
tile_u16 x, mask, k;
TEXPANDS(x,    (uint16_t)0x1234);   // prior dst value（就地 false-source）
TEXPANDS(mask, (uint16_t)0);        // 全假谓词
TEXPANDS(k,    (uint16_t)0xABCD);   // true-source（不应被选中）
TSEL(x, mask, k);                   // 就地: x = mask ? k : x_prior; mask=0 -> x 应保持 0x1234
TSTORE(gy, x);
```

**期望**：每元素 `x == 0x1234`。**实测**（gfrun）：

```
$ xxd -l 16 -g 2 output.bin
00000000: 0000 0000 0000 0000 0000 0000 0000 0000   ← 全 0x0000，而非 0x1234
```

对照：已有 `tsel_inplace_probe`（`mask=1`，测 `dst=trueSrc`）**通过**——因为它只走 true 分支、
**不触碰 prior-dst**；本探针 `mask=0` 专测 prior-dst 读取，暴露"读到 0"。

## 定位证据（模型侧）

在 `SoftCore::ExecuteTSEL`（`emulator/engine/TEPLEngine.cpp`）就地路径打印（`block->srcTile.size()==2`）：

```
[DIAG-TSEL] size=2 mask0=0x0  src0(true)=0x7f81  src1(false=dstTile[0])=0x0
```

- `mask0=0`（正确）、`src0(true)` 正确，但 **`src1`（false-source，来自 `dstTile[0]`）= 0**，
  而它应等于 dst 的先前值。
- 另一实验（把 finalize 砍到只留主路径、无 TSEL）证明 dst tile 里**确有**正确先前值；一旦插入
  这条本应 no-op 的 TSEL，先前值即丢失。

⟹ 结论：`1f398190` 让 execute 侧读 `dstTile[0]` 作 false-source，但**就地 lowering 没把 dst 的
先前值绑定到该目的寄存器**——TSEL 的 `->dst` 是一个新的 SSA 定义寄存器（fresh/0），而先前值仍在
旧寄存器里。模型读新目的寄存器 → 0。**根位置在 LinxV5 后端的就地 TSEL 寄存器分配（或就地 B.IOT
的 false-source 语义约定），而非 emulator 单点。**

## 与 cmode 无关（排除项）

一度怀疑是 `TCMP/TCMPS` 的 CmpMode 编解码错位，**已排除**：后端 `LinxV5BaseInfo.h` 的
`enum CmpMode{EQ=0,NE=1,LT=2,GT=3,LE=4,GE=5}` @ `Inst{31-29}`，与模型 `CMode` 枚举 + `(word>>29)&0x7`
**逐值同位**；`parseCmpMode` 对 token `.lower()` 后 `"eq"→EQ=0`。TCMPS 掩码本身正确，问题纯在 TSEL。

## 影响面

- `dynamic_mx_quant` 的 fp4 / cuBLAS 特殊值守卫 `finalize_recip_u16`（inf/zero/special 三 `TSEL`）
  依赖就地 TSEL 的 false-source 直通（`mask=0` 时保留主路径 recip）。当前缺陷下，**每个 no-op
  TSEL 都把 recip 清零** → data 量化全塌成 0（fp4 tail res_check：output 全 ±0.0）。
- 凡使用就地 `TSEL` 且依赖 false-source=prior-dst 的 kernel 均受影响；不影响 `mask` 全真的用法。

## 修复方向

1. **后端（首选）**：LinxV5 就地 `TSEL` lowering 须保证 `->dst` 目的寄存器 = 持有 false-source
   先前值的寄存器（真正就地），或在 B.IOT 编码里显式携带 false-source 源，使 emulator 可取到先前值。
2. **模型（配合）**：若采用显式 false-source 编码，`ExecuteTSEL` 相应从该源读取，而非 `dstTile[0]`。
3. 修复后本探针须输出 `0x1234`，且 `dynamic_mx_quant_tail_ocp_fp4` 的三守卫版 res_check 通过。

## 现状

- fp4 kernel **已按用户要求保留三守卫**于源码；模型含 `ab822e7a`+`1f398190` 崩溃修复但**不含**本
  prior-dst 语义修复，故带守卫版 res_check 仍 fail（output 全 ±0）；等价主路径（去 TSEL）可 pass。
- 参见 [[ISSUE_tsel_inplace_lowering]]（崩溃层）。
