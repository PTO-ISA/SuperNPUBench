#include <cstdint>

#include "benchmark.h"
#include "fileop.h"
#include "multi_thread_res_check.h"
#include "solution/transpose/transpose_nd_v3_2_1.hpp"

// v3.2.1 测试（Host-tiling）：shape/perm 以**运行时数组**传入，
// 覆盖三类场景与常见 rank/perm。
//   CASE=0 1D [4096]        perm(0)              -> TENSOR_MOVE
//   CASE=1 2D [64,64]       perm(1,0)            -> GATHER
//   CASE=2 3D [64,64,256]   perm(1,0,2)          -> N_LAST
//   CASE=3 3D [32,64,512]   perm(0,2,1)          -> GATHER
//   CASE=4 3D [16,32,64]    perm(2,0,1)          -> GATHER
//   CASE=5 4D [8,8,8,8]     perm(0,2,1,3)        -> GATHER
//   CASE=6 4D [1,16,936,128] perm(0,2,1,3)       -> N_LAST
//   CASE=7 5D [8,8,8,16,32] perm(0,1,4,3,2)      -> GATHER
#ifndef V32_TILE_ELEMENTS
#define V32_TILE_ELEMENTS 512
#endif
#ifndef V32_CASE
#define V32_CASE 0
#endif
#ifndef V32_PE
#define V32_PE 4
#endif

#if V32_CASE == 1
constexpr int kRank = 2;
constexpr int kCap = 64 * 64;
#elif V32_CASE == 2
constexpr int kRank = 3;
constexpr int kCap = 64 * 64 * 256;
#elif V32_CASE == 3
constexpr int kRank = 3;
constexpr int kCap = 32 * 64 * 512;
#elif V32_CASE == 4
constexpr int kRank = 3;
constexpr int kCap = 16 * 32 * 64;
#elif V32_CASE == 5
constexpr int kRank = 4;
constexpr int kCap = 8 * 8 * 8 * 8;
#elif V32_CASE == 6
constexpr int kRank = 4;
constexpr int kCap = 1 * 16 * 936 * 128;
#elif V32_CASE == 7
constexpr int kRank = 5;
constexpr int kCap = 8 * 8 * 8 * 16 * 32;
#elif V32_CASE == 9
constexpr int kRank = 3;
constexpr int kCap = 200 * 200 * 32;
#else
constexpr int kRank = 1;
constexpr int kCap = 4096;
#endif

namespace {
alignas(4096) std::int32_t input[kCap];
// 轻量 PE 屏障：由 PE0 发布，其余 PE 自旋等待。
// 仅在 `multiThreadNum == PE` 时使用；带自旋上限，参数不匹配时退化为"继续执行"而非死锁。
namespace {
volatile int g_bar_init_arrive = 0;
volatile int g_bar_init_release = 0;
inline void pe_barrier_init(int tid, int npe) {
    if (npe <= 1) return;
    __atomic_add_fetch(&g_bar_init_arrive, 1, __ATOMIC_ACQ_REL);
    if (tid == 0) {
        for (long s = 0; s < (1L << 26); ++s) {
            if (__atomic_load_n(&g_bar_init_arrive, __ATOMIC_ACQUIRE) >= npe) break;
        }
        __atomic_store_n(&g_bar_init_release, 1, __ATOMIC_RELEASE);
    } else {
        for (long s = 0; s < (1L << 26); ++s) {
            if (__atomic_load_n(&g_bar_init_release, __ATOMIC_ACQUIRE) != 0) break;
        }
    }
}
}  // namespace

alignas(4096) std::int32_t output[kCap];
#ifdef RES_CHECK
MultiThreadResCheckSync res_check_sync{};
#endif
}  // namespace

int main() {
    const std::uint32_t tid = get_thread_idx();
    // 运行时 shape/perm（模拟动态传入）。
    // NOTE: 声明为 volatile —— 本模型会把相邻的 32 位标量写合并成一次 64 位栈写
    //       并丢掉高半（曾导致 shape 出现 0）；volatile 强制逐元素写。
    volatile std::uint32_t rt_in[kRank];
    volatile std::uint32_t rt_out[kRank];
    volatile std::uint32_t rt_perm[kRank];
    // 同理，不走 constexpr 常量数组（其部分元素会被错误物化，见 issue #1038 同类问题）。
#if V32_CASE == 0
    rt_in[0] = 4096; rt_perm[0] = 0;
#elif V32_CASE == 1
    rt_in[0] = 64; rt_in[1] = 64; rt_perm[0] = 1; rt_perm[1] = 0;
#elif V32_CASE == 2
    rt_in[0] = 64; rt_in[1] = 64; rt_in[2] = 256;
    rt_perm[0] = 1; rt_perm[1] = 0; rt_perm[2] = 2;
#elif V32_CASE == 3
    rt_in[0] = 32; rt_in[1] = 64; rt_in[2] = 512;
    rt_perm[0] = 0; rt_perm[1] = 2; rt_perm[2] = 1;
#elif V32_CASE == 4
    rt_in[0] = 16; rt_in[1] = 32; rt_in[2] = 64;
    rt_perm[0] = 2; rt_perm[1] = 0; rt_perm[2] = 1;
#elif V32_CASE == 5
    rt_in[0] = 8; rt_in[1] = 8; rt_in[2] = 8; rt_in[3] = 8;
    rt_perm[0] = 0; rt_perm[1] = 2; rt_perm[2] = 1; rt_perm[3] = 3;
#elif V32_CASE == 6
    rt_in[0] = 1; rt_in[1] = 16; rt_in[2] = 936; rt_in[3] = 128;
    rt_perm[0] = 0; rt_perm[1] = 2; rt_perm[2] = 1; rt_perm[3] = 3;
#elif V32_CASE == 9
    rt_in[0] = 200; rt_in[1] = 200; rt_in[2] = 32;
    rt_perm[0] = 1; rt_perm[1] = 0; rt_perm[2] = 2;
#else
    rt_in[0] = 8; rt_in[1] = 8; rt_in[2] = 8; rt_in[3] = 16; rt_in[4] = 32;
    rt_perm[0] = 0; rt_perm[1] = 1; rt_perm[2] = 4; rt_perm[3] = 3; rt_perm[4] = 2;
#endif
    for (int i = 0; i < kRank; ++i) rt_out[i] = rt_in[rt_perm[i]];
    // volatile → 普通数组转一手，供 `MakeV32Tiling()`（const 指针参数）使用。
    std::uint32_t fin[kRank], fout[kRank], fperm[kRank];
    for (int i = 0; i < kRank; ++i) {
        fin[i] = rt_in[i];
        fout[i] = rt_out[i];
        fperm[i] = rt_perm[i];
    }
    // 自初始化：所有 PE 写入相同数据（确定性），供 host 侧 golden 比对；
    // 注意：这两段循环会被算进 gfsim 的 Total Cycles，纯性能测量请加 -DPERF_ONLY 跳过。
#ifndef PERF_ONLY
    // 只由 PE0 初始化，再同步到所有 PE，避免"慢 PE 把已完成结果重置为 -1"
    if (tid == 0) {
        for (int i = 0; i < kCap; ++i) input[i] = i;
        for (int i = 0; i < kCap; ++i) output[i] = -1;
    }
    pe_barrier_init(static_cast<int>(tid), V32_PE);
#endif
#ifdef RES_CHECK
    res_check_publish_inputs(res_check_sync, tid);
#endif
    // 与 ops-math 原算子对齐：所有 case 统一走 Host-tiling（HT）路径。
    const auto tiling = supernpu::tile_isa::v321_detail::MakeV32Tiling(
        kRank, fin, fout, fperm, static_cast<std::uint32_t>(kCap), V32_TILE_ELEMENTS,
        static_cast<int>(sizeof(std::int32_t)));
    BENCHSTART;
    supernpu::tile_isa::transpose_nd_v3_2_1_dispatch<std::int32_t, kCap, V32_TILE_ELEMENTS, V32_PE>(
        input, output, tiling, static_cast<int>(tid));
    BENCHEND;
#ifdef RES_CHECK
    res_check_wait_for_all(res_check_sync, tid);
#endif
#ifdef ENABLE_BINARY_OUTPUT
    if (tid == 0) {
        writeBinaryFile(CHK_DIR "/output.bin", reinterpret_cast<std::uint8_t *>(output),
                        sizeof(std::int32_t) * static_cast<std::size_t>(kCap));
    }
#endif
    return 0;
}
