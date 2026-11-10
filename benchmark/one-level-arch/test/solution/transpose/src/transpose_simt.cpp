#include <cstdint>

#include "benchmark.h"
#include "fileop.h"
#include "solution/transpose/transpose_nd_simt_elementif.hpp"

// element-if（SIMT/SMALL_SHAPE）测试：4D [8,8,8,8] perm(0,2,1,3)
#ifndef SIMT_PE
#define SIMT_PE 4
#endif
#ifndef SIMT_CASE
#define SIMT_CASE 0
#endif

#if SIMT_CASE == 1
constexpr int kRank = 3;
constexpr int kNumel = 32 * 64 * 512;
#else
constexpr int kRank = 4;
constexpr int kNumel = 8 * 8 * 8 * 8;
#endif

namespace {
alignas(4096) std::int32_t input[kNumel];
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

// 轻量 PE 屏障：由 PE0 发布，其余 PE 自旋等待。
// 仅在 `multiThreadNum == PE` 时使用；带自旋上限，参数不匹配时退化为"继续执行"而非死锁。
namespace {
volatile int g_bar_done_arrive = 0;
volatile int g_bar_done_release = 0;
inline void pe_barrier_done(int tid, int npe) {
    if (npe <= 1) return;
    __atomic_add_fetch(&g_bar_done_arrive, 1, __ATOMIC_ACQ_REL);
    if (tid == 0) {
        for (long s = 0; s < (1L << 26); ++s) {
            if (__atomic_load_n(&g_bar_done_arrive, __ATOMIC_ACQUIRE) >= npe) break;
        }
        __atomic_store_n(&g_bar_done_release, 1, __ATOMIC_RELEASE);
    } else {
        for (long s = 0; s < (1L << 26); ++s) {
            if (__atomic_load_n(&g_bar_done_release, __ATOMIC_ACQUIRE) != 0) break;
        }
    }
}
}  // namespace

alignas(4096) std::int32_t output[kNumel];
}  // namespace

int main() {
    const std::uint32_t tid = get_thread_idx();
#ifndef PERF_ONLY
    // 只由 PE0 初始化并发布（同 transpose_v32）
    if (tid == 0) {
        for (int i = 0; i < kNumel; ++i) input[i] = i;
        for (int i = 0; i < kNumel; ++i) output[i] = -1;
    }
    pe_barrier_init(static_cast<int>(tid), SIMT_PE);
#endif
    BENCHSTART;
#if SIMT_CASE == 1
    supernpu::tile_isa::transpose_nd_simt_elementif_tile<
        std::int32_t, 3, kNumel, 128, std::array<int, 8>{32, 64, 512}, std::array<int, 8>{0, 2, 1}>(
        input, output, static_cast<int>(tid), SIMT_PE, tid == 0);
#else
    supernpu::tile_isa::transpose_nd_simt_elementif_tile<
        std::int32_t, 4, kNumel, 128, std::array<int, 8>{8, 8, 8, 8}, std::array<int, 8>{0, 2, 1, 3}>(
        input, output, static_cast<int>(tid), SIMT_PE, tid == 0);
#endif
    BENCHEND;
    // 所有 PE 到齐后再导出，避免 PE0 提前 dump 到未完成的 tile
    pe_barrier_done(static_cast<int>(tid), SIMT_PE);
#ifdef ENABLE_BINARY_OUTPUT
    if (tid == 0) {
        writeBinaryFile(CHK_DIR "/output.bin", reinterpret_cast<std::uint8_t *>(output),
                        sizeof(std::int32_t) * kNumel);
    }
#endif
    return 0;
}
