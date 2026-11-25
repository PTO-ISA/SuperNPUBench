# main 复查问题修复

修复基线为 `4cd779c3d08a84d0e4c42e4aea42b2ca75ff139f`。未修改编译器或模拟器源码。

## 修复内容

1. **Packed matmul stride**：GM tensor shape/stride 使用逻辑元素数；只有指针偏移使用 packed 字节数。修正 Shared、reuseA、reuseA full-cache 三条路径，覆盖 HiF4/MXFP4，普通 8-bit/16-bit/32-bit stride 不变。
2. **HiF4 FA 增量构建**：共用 fa_hif4.o 增加强制重编依赖，切换入口、shape、RES_CHECK 或修改 header 时不会链接旧配置的对象。其他 FA 入口保持现有构建行为。
3. **Matmul hosted checker**：默认使用 `/Users/blacktraker/Programming/gitproj/DV4/SuperScalarModel/bin/gfrun`；golden-check 编译 `CHK_DIR=.`，为每个 ELF 提供独立 hosted-file-root，并从模型工作树运行。校验正式完成状态；取消预填零输出，允许真实全零结果，缺失输出仍失败。
4. **宏反汇编兼容**：microbenchmark 和 one-level 的 make diss 共用 `/Users/blacktraker/Programming/gitproj/DV4/SuperNPUBench/benchmark/one-level-arch/test/common/dump_elf.py`。仅针对已知 `no complete macro instruction form` 错误尝试物理反汇编；保留原诊断并写 mode sidecar，其他错误或物理反汇编失败仍返回非零。fixp 报告支持物理 BSTART 格式，并从原始指令字解码 FPATR。

第 4 项是仓库侧兼容处理，不代表上游 llvm-objdump 宏渲染缺陷已经修复。

## 验证结果

| 检查 | 结果 |
|---|---|
| Matmul precision-check：FP32/FP16/BF16/MXFP8/MXFP4/HiF4 | 6/6 数值通过 |
| HiF4 B2/M256/N128/K256/tM128/tN64/tK128 | 65,536/65,536 精确一致，覆盖 batch、M/N tile、K chain |
| MXFP4 reuseA M64/N64/K64 与 M128/N64/K64 | 非 full-cache 路径均精确一致 |
| MXFP4 reuseA M128/N128/K128/tK64 | full-cache 路径 16,384/16,384 精确一致 |
| HiF4 FA 同一 OBJ_ROOT 依次普通 make fixpipe、kchains、2d（最后 Skv 改 256） | 三次均实际重编，三组数值检查通过；未依赖 make -B |
| Local/Shared S8_ROWMAX make diss | 返回成功，物理 fallback 和原始诊断可见；FPATR/aux/IOS 契约检查通过 |
| Local/Shared S8_ROWMAX gfrun | 正式完成且 status=0 |
| one-level HiF4 FA make diss | 成功，验证同一兼容 helper 在 one-level 的接入 |
| Python 回归测试 | 7/7：宏成功、已知失败降级、无关失败保留、物理失败保留、两种反汇编格式、零/缺失输出、hosted root 与完成状态 |

固定编译器为 `/Users/blacktraker/Programming/gitproj/DV4/linx-toolchain-build/output/linx_blockisa_llvm_musl/bin`。功能检查使用 `/Users/blacktraker/Programming/gitproj/DV4/SuperScalarModel/bin/gfrun`，cwd 为 `/Users/blacktraker/Programming/gitproj/DV4/SuperScalarModel`。

## 复现

```bash
export COMPILER_DIR=/Users/blacktraker/Programming/gitproj/DV4/linx-toolchain-build/output/linx_blockisa_llvm_musl/bin
make -C /Users/blacktraker/Programming/gitproj/DV4/SuperNPUBench/benchmark/one-level-arch/test/kernel/matmul precision-check OBJ_ROOT=/tmp/issue298-fix-check
make -C /Users/blacktraker/Programming/gitproj/DV4/SuperNPUBench/benchmark/one-level-arch/test/kernel/matmul golden-check TESTCASE=matmul_lowp LP_MODE=HIF4X2 B=2 M=256 N=128 K=256 tM=128 tN=64 tK=128 OBJ_ROOT=/tmp/issue298-fix-hif4
make -C /Users/blacktraker/Programming/gitproj/DV4/SuperNPUBench/benchmark/one-level-arch/test/kernel/matmul golden-check TESTCASE=matmul_lowp LP_MODE=MXFP4 LOWP_REUSE_A=on B=1 M=128 N=128 K=128 tM=128 tN=64 tK=64 OBJ_ROOT=/tmp/issue298-fix-reuse
make -C /Users/blacktraker/Programming/gitproj/DV4/SuperNPUBench/microbenchmark/fixp FIXP_MODE=S8_ROWMAX res_check=on OBJ_ROOT=/tmp/issue298-fix-diss diss
python3 /Users/blacktraker/Programming/gitproj/DV4/SuperNPUBench/benchmark/one-level-arch/test/common/test_dump_elf.py
python3 /Users/blacktraker/Programming/gitproj/DV4/SuperNPUBench/benchmark/one-level-arch/test/kernel/matmul/src/test_gfrun_matmul.py
```

详细本轮日志保存于 `/Users/blacktraker/Programming/gitproj/DV4/SuperNPUBench/outputs/issue298-main-fixes-20261125`。
本轮为已知问题的针对性验证，未重跑全仓功能或 gfsim 性能回归。
