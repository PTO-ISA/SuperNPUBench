# tileop-guard —— PTO-ISA microbench API 精度看护

对 PTO-ISA microbench（TileOP-API intrinsic）做**精度看护**：不仅验"能不能跑"（gfrun 通过），更验
"算得对不对"——用 host 侧独立 golden 逐元素对照 **pto-spec 规范 ASL 定义的预期语义**。

> 与 version-tracking / pr-tracking 的区别：那两个只看 gfrun/gfsim 是否 PASS；本套看逐元素数值正确性。

## 判据：四态层层递进

每个 case 停在能到达的最高层：

1. **编译失败**（compile-fail）：工具链拒绝（header↔backend skew、后端缺 pattern 等）。
2. **执行失败**（run-fail）：编译过、gfrun 崩（模型契约拒绝 / 未实现）。
3. **精度失败**（precision-fail / witness）：跑通但输出与 golden 不符。golden 钉 pto-spec **预期**语义，
   实现背离自然落此层 = **忠实暴露缺口，非回归**。
4. **精度正确**（PASS）：功能 + 精度双看护通过。

`run-only`：无可校数值输出的终态（纯搬运 tload/tstore/tmov/tprefetch）。

## golden 纪律（关键）

- golden 是 **host 侧独立 numpy oracle**，仅从**接口语义**（pto-spec 规范 ASL + intrinsic 文档）实现，
  **不读模型实现** → 真正独立的参照。
- golden 必须钉 **pto-spec 规范定义的预期语义**（写前先核实、别假设、别拟合模型观测行为）——否则会把
  模型 bug 焊进 oracle，让看护对缺陷视而不见。
- 输入由 host（numpy）生成并落盘（设备端填充会被 tile 后端错编）；ELF 整块 `read()` 读入。

## 目录结构

| 路径 | 说明 |
|---|---|
| `vec/ sfu/ cube/ fixp/ tlsu/ misc/` | 各域 demo（`src/*.cpp` + 每域 `Makefile`） |
| `golden/golden.py` | host 独立 golden：`gen` 造输入 / `check` 校输出，`REG[case]` 注册每 case 语义 |
| `common/` | 共享驱动模板 `guard_common.hpp` / `guard_case.hpp` + `guard_io`（无 printf 落盘） |
| `retired/` | 已被 PTO-SPEC ADR 退休的直接 Tile op（隔离出活跃看护，fail-closed） |
| `run_guard.sh` `env.sh` `Makefile.common` | 执行入口 |
| `compare/` `output/` | 运行产物（gitignore，非提交） |

## 执行方式

```bash
# 1. 指向 linx 工具链 + gfrun：export LINX_ROOT=<含 linx-toolchain-build/ 和 SuperScalarModel/ 的目录>
#    （或直接 export COMPILER_DIR / GFRUN），再 source env.sh 生效
export LINX_ROOT=/path/to/your/linx-root
source benchmark/one-level-arch/test/solution/tileop-test/env.sh          # 设 COMPILER_DIR / GFRUN

# 2. 跑单个 case
bash benchmark/one-level-arch/test/solution/tileop-test/run_guard.sh <domain> <case>
#   例: bash benchmark/one-level-arch/test/solution/tileop-test/run_guard.sh sfu tcolmax

# 3. 跑整个域（domain ∈ vec/sfu/cube/fixp/tlsu/misc）
bash benchmark/one-level-arch/test/solution/tileop-test/run_guard.sh <domain>
```

**res_check 精度流程（每 case）**：
`golden.py gen`（造 `CHK_DIR/in_*.bin`，host 拥有输入）→ `make`（编译 ELF）→ `gfrun`（读 `in_*`、dump `out.bin`）
→ `golden.py check`（numpy 独立逐元素对照，rc0=PASS / 非0=MISMATCH）。

**输出列**：`compile | gfrun | precision`，落为四态 `compile-fail / run-fail / precision-fail / pass`
（未注册 golden 的 case 显示 `run-only`）。失败 case 另落 `compare/<域>/<case>/{compile.log,gfrun.log}` +
一条 `ERRSIG` token（报错点）。

## 注意

- **改共享头（`common/*.hpp`）后必须 clean 重编**：`run_guard.sh` 增量 make 只看 `.cpp`，旧 `.o` 会假崩/假过。
- 判定前统一走同一套 linx 工具链 + gfrun；换工具链须干净重编。
- golden 涉舍入/饱和/默认值时，必须回 pto-spec ASL 核实规范默认（别信经验拟合）。
