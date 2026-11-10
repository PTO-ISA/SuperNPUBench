# FA MXFP4 AlgoD D128（algod2 / unscoped-D2）归档（2026-11-10）

**范围**：这是 `algod2.elf` 这一颗二进制的算子侧归档——源码快照、构建配方、ELF/diss 身份、以及它在 4 PE 模型上的两组测量记录（本分支 143,883 / 主线同配置 143,749）。**不替换** `eval/perf_data` 里的任何归档，也不改变默认算子入口。

**结论（一句话）**：这颗 ELF 的测量值对配置高度敏感——同一 ELF、同一份入口配置下，**主线 `34c110137` 是 143,749 拍，而带本轮模型改动（#1182 等）的分支是 143,883 拍**，即本分支在该口径下**不快于**主线。本包因此只声明"在同一分支内，拆分器跨类头阻修复让深队列病理消失"，不声明对主线的收益。

| case | 模型 | 有效配置差异 | Total Cycles |
|---|---|---|---:|
| `fa-64k-unscoped-d2-258099`（同 ELF，已有记录，不复制） | SuperScalarModel-asl `bdbafed5` | 早期 unscoped 配置 | 258,099 |
| `algod2-d128-143883`（本包主例） | SuperScalarModel `dd83d4519` | tag=36、类感知准入、`vecL1Depth=32` | **143,883**（两次，逐指标相同） |
| `algod2-d128-143749-mainline`（对照） | SuperScalarModel 主线 `34c110137` | tag 被夹到 8、无类感知准入 | 143,749 |
| `algod2-d128-144036-prefix`（分支内对照） | 同分支更早状态 | 未含 #1182 | 144,036 |
| `algod2-lazy128-165091`（病理，仅 trace） | 同分支更早状态 | lazy + `vecL1Depth=128` | 165,091（无成功 log） |

> 258,099 与 143,883 是**同 ELF、不同模型与配置**，不可相减当收益；详见 `operator_identity.json` 的 `cross_record_rules`。

## 复现（不需要编译器）

1. 取 ELF：`fa-operator-r2.tar.gz` 里的 `bin/fa-64k-unscoped-d2-258099.elf`（sha256 `3dba8965…`），或直接使用模型仓 `repro/algod2/algod2.elf`（字节相同）。
2. 准备模型：`LinxISA/SuperScalarModel` 分支 `repro/algod2-final-143883`，commit `dd83d4519`，构建 `bin/gfsim`（本包实测二进制 sha256 `69ed905c…`）。
3. 把 `configs/fa_fp4_tag36_brob1024_veciq32_disp4.conf` 放到模型仓的 `configs/`（`--conf` 会解析为 `configs/<name>`）。

```bash
./bin/gfsim -f algod2.elf --conf fa_fp4_tag36_brob1024_veciq32_disp4
# 期望：Cube Total Cycles = 143883
```

主线对照：`git checkout 34c110137` 后重建，用同一份 conf 运行，期望 143,749。

## 这颗 ELF 的两套源码不要混

- `source/`（本包）：**unscoped D2**，`fa_lowp_algD2.hpp` = `1bc9534a…`（28,523 B），是编译出 `3dba8965…` 的那一版；构建宏含 `-DFA_ALGD2=1`、`-DFA_ALGB_D_DBLQ=1`。它此前只存在于 `fa-operator-r2.tar.gz`，现在随本包进入 git 树。
- 姊妹目录 `fa-scoped-packed-rowsum-20261109/source/`：**scoped D2**，`fa_lowp_algD2.hpp` = `203a0ee7…`（28,799 B），**不是**这颗 ELF 的来源。按那份源码重建会得到另一个二进制。

## 数值状态

**未验证**。所有记录的日志只有运行完成证据（`Report Stop` + Syscall Barrier），没有 PASS 或 golden 比对；算子侧 `release_index.json` 对该形状同样标注 `zero-input timing only`。因此本包只提供计时，不提供功能结论。

## 缺项

见 `delivery_index.json` 的 `missing`：

- 模型侧 log 未复制到本仓（按职责分工留在模型仓，本包记录路径与 sha256）；
- 历史 165,091 / 143,996 只有 swimlane、没有成功 log；
- 本配置未采 PipeView；完整 swimlane 未随包（在模型仓，64 MB 压缩）；
- 未从 Release 重新下载并做隔离目录复跑；
- 无独立复核；主线同配置的差值未做分项归因。

## 关联

- 模型侧归档（同一批材料的另一本账）：`LinxISA/SuperScalarModel` PR #1240 / `archives/ndf-algod2-20261110-r1`
- 机制 issue：#1175；修复 PR：#1182（拆分器类感知准入）、#1219（resident tag 上限）
- 同一 ELF 的历史记录：`fa-scoped-packed-rowsum-20261109` 的 `fa-64k-unscoped-d2-258099-portable-r2`（258,099）