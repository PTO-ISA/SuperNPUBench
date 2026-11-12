# 负控与失败路径（NDF §6.2）

## 1. 比较器负控（A/B，唯一差异是 golden 的 1 个字节）

- 被测入口：r1 的 64-lane SoA check ELF
  （`outputs/hash64_batchmap_20261105/src/check.elf`，sha256 `cec67d18…`）。
- 方法：fixture 从参考 ELF 的 `0x26f6000` incbin 24 KiB golden；该 blob 在
  `check.elf` 中只出现一次（`0x3a6e000`）。把首字节 XOR 0xFF 得到 bad ELF
  （sha256 `00eeb109…`），其余字节完全相同。
- 运行（同一 gfsim、同一配置、`--test-finisher 1`）：

| ELF | finisher 状态 | gfsim 退出码 | 日志 |
|---|---|---|---|
| `check.elf`（正常） | `val=0x5555 pass` | 0 | `r1check_ok.log`（已脱敏；原件 sha256 `0b484e3f…`，派生 `57980c12…`） |
| `r1check_bad.elf`（golden 首字节损坏） | `val=0x3333 fail` | 70 | `r1check_bad.log` |

结论：错误输出被比较器拒绝，且 runner 返回非零；两次运行的 `Total Cycles` 都是
273,152（这是含校验的 check 运行，不是性能口径）。

复现：`patch_golden.py <reference-elf> <check.elf> <bad.elf>`。

注：`r1check_ok.log` 首行的个人路径已替换为 `<REPO>` / `<GFSIM_WORKTREE>`；
`r1check_bad.log` 的首行只有 `/tmp/negctl`，无需脱敏。

## 2. 缺失输入的拒绝

| 场景 | 命令 | 实际结果 |
|---|---|---|
| ELF 不存在 | `MODEL=… ./build/run_perf.sh /tmp/does_not_exist.elf random300` | rc=2，`missing ELF: …`（`missing_elf.log`） |
| `MODEL` 未设置 | `env -u MODEL ./build/run_perf.sh <elf> random300` | rc=1，`MODEL: set MODEL …`（`missing_model.log`） |
| 编译器变量未设置 | `env -u CLANG ./build/build_kernel.sh matrix 64 2_2 <out>` | rc=1，`CLANG: set CLANG …`（`missing_clang.log`） |

## 3. checker / 模型身份

| 项 | 值 |
|---|---|
| check ELF | sha256 `cec67d18255baf922301494255d462368209c4cacc4ff7ea464fa1101e05be16`，61,302,360 B |
| golden 来源 | 参考 ELF `960ba161…` 偏移 `0x26f6000`，24,576 B |
| fixture | `outputs/hash64_batchmap_20261105/src/fixture.cpp`（incbin golden/keys/values） |
| 模型 | `SuperScalarModel-pr1177` @ `18b24f6ca`，gfsim sha256 `12066cbd…` |
| 编译器 | clang 15.0.4（linx64v5），sha256 `35898d12…`（check.elf 构建用） |
| 运行命令 | 见 `r1check_ok.log` 首行（random300 + `--test-finisher 1`） |

## 4. 未覆盖的负控

- 算子 AoS verify ELF（r3）在 gfsim 上触发既有断言
  `responseReturnBeatId == geo.mrobEntryIdx`（`GAP-R3-1`），无法作为负控载体；
  其比较器负控保持 `not_checked`。
- 功能性模型（gfrun）当前工作树不在归档环境内，且主线 gfrun 拒绝该 ELF
  （`Cannot write byte to 0x00010040` / `ParseError`）；本归档不宣称已完成
  gfrun 维度的负控。
- “空 case / 未知 case / 零执行”对本归档不适用：入口是固定 6144 query 的
  kernel，没有可传空的 case 列表；理由记录在 `requirements_coverage.md`。
