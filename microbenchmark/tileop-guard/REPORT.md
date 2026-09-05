# TileOP-API v0.58 文档看护报告 — 索引

看护结果**按时间戳分文件**，各文件诚实记录该时点的现状快照。

| 时间戳 | 文件 | 判据 | 结果 |
|--------|------|------|------|
| 2026-09-01 | [`REPORT_20260901.md`](REPORT_20260901.md) | 三态 | 125 = 52 精度PASS / 47 run-only / 12 编译失败 / 14 run-fail;据此提 docs/gfrun/linx 三份 issue |
| 2026-09-03 | [`REPORT_20260903.md`](REPORT_20260903.md) | 四态 | 126 = 101 精度正确 / 11 精度失败(witness) / 4 run-only / 2 编译失败 / 8 run-fail |
| 2026-09-05 | [`REPORT_20260905.md`](REPORT_20260905.md) | 四态 | 工具链升 ops-20260904;125 已发布 case = 77 精度正确 / 12 精度失败 / 4 run-only / 3 编译失败 / 29 run-fail;4 类缺口自证入 issue(B.FPATR None matmul/fixp、行列归约 skew、TINVALID selector、tgpr2t 后端);mgather_cas 证为 demo bug 已修 PASS |

- **三态 → 四态**:2026-09-03 起把「精度失败」独立成层。golden 改钉 pto-spec **预期语义**,实现
  背离自然落精度失败,作模型缺口 witness。详见各文件开头判据说明。
- **统计范围 = 已发布算子**;header 标 `(unreleased)` 的算子允许出错,不纳入统计/issue。
- **每条报错先自证「非 demo 写法错」再归因**:用 release 自带参考源现编复现 / model 源码定位;
  未能自证者不计 witness、不入 issue(本轮据此把 mgather_cas 从 witness 降为 demo bug 并修复)。
- issue 清单见 `issues/`(docs / gfrun / linx + 各时间戳增量;2026-09-05 新增 `ISSUE_gfrun_ops20260904_model_gaps.md`、`ISSUE_linx_ops20260904_toolchain_gaps.md`)。
