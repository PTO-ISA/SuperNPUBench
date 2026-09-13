#include "multi_thread/quant/dynamic_hi_f4_quant/dynamic_hi_f4_quant_tail.h"
#include <cstdint>

// ============================================================================
// dynamic_hi_f4_quant 尾轴 encode —— 路线 A「整体流程框架」发射见证 (emit witness)
// ============================================================================
// 按用户 2026-08-14 决策实现的框架（见 kernels/.../dynamic_hi_f4_quant_tail.h 头注）：
//   「先按照算法实现整体流程框架，绕过不可实现的部分，代码中预留位置，待后续与硬件商讨
//    适配方案。同时如实记录问题」。
//
// 本文件不是数值 harness——框架含 4 个 `_PLACEHOLDER` 缺口 helper（RECORD 问题3/4/10），
// 其数值输出**不可信**。目的仅为：在树内以 `make ... TYPE=FRAMEWORK diss` 复现
//   (1) 框架**整体编译干净**（EXIT=0），
//   (2) 反汇编含真实 BSTART.TEPL 块（TABS/TROWMAX/TROWEXPANDMUL/TMAXS/TCMP/TSEL/TCVT/
//       TROWSUM/TSHL/TMUL/TCI/TSTORE… 均真实发射）。
//
// ⚠️ 验证边界（RECORD 头）：工具链↔仿真器 skew 下新编 ELF 无法稳定 gfrun/gfsim，故本见证
//    是 **op-review + 编译/反汇编级别**，无运行期数值验证。占位步骤替换前数值不可信。
//
// 覆盖点（编译期实例化，穿透 kernel 的每条 if constexpr / full+tail 双 pass）：
//   - InT=__bf16（直载）与 InT=__half（TCVT→bf16 分支，DESIGN §0）；
//   - M=10 → TileM=8：full_m=1 满行 pass + M_tail=2 尾行 pass（ValidRow 收窄）都实例化；
//   - N=128 → numKb=2：多 block 迭代。
//
// 关键规避已固化在框架头（RECORD 问题11，本见证的编译干净即其证据）：所有 bf16 标量走
// `__builtin_bit_cast(__bf16, 位模式)`（GE 阈值为编译期模板参数），bf16 0 tile 用 TSUB(t,t)——
// 否则 LinxV5 后端在 float→bf16 处 `UNREACHABLE (LegalizeIntegerTypes.cpp:57)` 崩溃、
// 或 bf16 0.0 落 zero 寄存器致 `B.IOR [zero],[]` 匹配失败。
// ============================================================================
using namespace supernpu::tile_isa::hif4quant;

static constexpr int M  = 10;   // → TileM=8, full_m=1, M_tail=2（两 pass 都实例化）
static constexpr int N  = 128;  // → numKb=2（多 block）

static __bf16       xb[M * N]     __attribute__((aligned(4096))) = {};
static __half       xh[M * N]     __attribute__((aligned(4096))) = {};
static __fp4_e1m2x2 y [M * N / 2] __attribute__((aligned(4096))) = {};
static uint32_t     s [M * (N / 64)] __attribute__((aligned(4096))) = {};

int main() {
    // bf16 输入路径（直载）
    dynamic_hi_f4_quant_tail<M, N, 64, __fp4_e1m2x2, __bf16>(xb, y, s);
    // fp16 输入路径（half → bf16 TCVT 分支，DESIGN §0）
    dynamic_hi_f4_quant_tail<M, N, 64, __fp4_e1m2x2, __half>(xh, y, s);
    return 0;
}
