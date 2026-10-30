#pragma once

// ============================================================================
// mt_barrier.hpp — 多 PE 驱逐屏障 + 定向行驱逐工具（跨 PE 可见性原语）
//
// 移植自 kernels/solution/group_token_vec/group_token_vec_mt.hpp 的 mtBarrier
// （该实现在 gfsim fourpe 下实证 PASS，Total Cycles 1,282,647）。要点：
//   - 时序模型跨 PE 无 L1D snoop：flag 行须主动驱逐使 L1D miss 才能看到
//     新值；裸热自旋在低流量窗口存在确定性活锁（dyn 开发期实证）。
//   - 写方到达驱逐：写方不自旋，其脏 flag 行须立即定向驱逐一次，L2 尽早
//     可见（等待方的稀疏驱逐 poll 才有新值可读）。
//   - 等待方：寄存器驻留延迟链（零内存流量）+ 1/64 轮稀疏定向集驱逐 +
//     跳自槽（消除同地址 store→load 对）。
//   - 驱逐流量必须稀疏，过密会饿死 L2 的 tile 派发（prior 洪泛）。
//   - gfrun 无缓存，语义同原始屏障。
//
// 本文件额外提供 evict_lines()：对任意 GM 区间按 64B 行做同 set 定向
// 驱逐（读 16KB 对齐驱逐区的 5 个同 set 别名行）。用途：
//   - 标量写后的写方驱逐（脏行写回 L2，跨 PE 可见前提）；
//   - 标量读前的读方驱逐（丢弃可能陈旧的 L1D 行，强制经 L2 重取）。
//
// 差异（相对 group_token 原版）：tid/peCount 改为显式参数（不依赖
// get_thread_idx 的头文件链）；数组置于 mt_sync 命名空间（避免与
// group_token 的同名 static 符号冲突）。v1 按设计方案不复用/去重
// group_token 侧实现。
// ============================================================================

#include <cstddef>
#include <cstdint>

namespace supernpu {
namespace multi_thread {
namespace mt_sync {

// flag 行显式 64B 对齐；驱逐区 16KB 对齐承载（+16KB slack，运行期取
// 对齐基址——[实测] 本工具链对 alignas(16384) 静默截为段对齐 8，
// llvm-readelf 实证，alignas(64)/alignas(4096) 正常保留）。
//
// [set 索引实测推导] L1D set = addr[15:8]（bitList=[8..15]，bit 6/7 为
// cluster dispatch 位，见 l1.toml dcache_nset=256 / l1_disp_bit_arr）——
// 同 set 别名步距为 64KB 而非 16KB；且替换为 shift-LRU（refill 恒逐出
// way[n-1]，updateLRU 移 way0）：4-way 下需 5 次同 set 填充才能逐出任
// 意行（含刚轮询的 MRU flag 行）。故驱逐读取 64KB 步距 5 别名
// （group_token 原版的 16KB 步距每轮仅 1 次同 set 填充，逐出不可靠，
// 依赖时序侥幸）。驱逐区覆盖 5*64KB + 16KB slack = 352KB。
inline volatile std::uint32_t s_evict_span_raw[22 * 4096];
alignas(64) inline volatile std::uint32_t s_phase_done[4];

// 16KB 对齐的驱逐区基址（每次调用重取；数组基址为只读常量）。
inline volatile std::uint32_t *evict_base() {
    const std::uint64_t a =
        reinterpret_cast<std::uint64_t>(s_evict_span_raw);
    return reinterpret_cast<volatile std::uint32_t *>(
        (a + 16383u) & ~static_cast<std::uint64_t>(16383));
}

inline void compiler_barrier() { __asm__ volatile("" : : : "memory"); }

// ----------------------------------------------------------------------------
// mt_barrier：单行 flag + 写方到达驱逐 + 寄存器延迟链 + 1/64 稀疏驱逐自旋。
// 前置：tid < peCount <= 4；phase 单调递增（每次调用 +1，与 group_token 的
// 1..4 相同用法）。
// ----------------------------------------------------------------------------
inline void mt_barrier(std::uint32_t phase, std::uint32_t tid,
                       std::uint32_t peCount) {
    compiler_barrier();
    s_phase_done[tid] = phase;
    compiler_barrier();
    // 写方到达驱逐：立即将本 PE 的脏 flag 行定向驱逐一次，L2 尽早可见。
    // 64KB 步距 5 别名 = 5 次同 set 填充 ⇒ shift-LRU 必逐出 flag 行。
    const std::uint32_t word_off_w =
        (static_cast<std::uint32_t>(
             reinterpret_cast<std::uint64_t>(&s_phase_done[0]) >> 2)) &
        4095u;
    for (std::uint32_t k = 1; k <= 5; ++k) {
        (void)evict_base()[k * 16384u + word_off_w];
    }
    compiler_barrier();
    // flag 行所在 L1D set 的字偏移：驱逐区基址 16KB 对齐 →
    // s_evict_span[k*4096 + word_off] 与 s_phase_done[0] 同 set（k 任意）。
    const std::uint32_t word_off =
        (static_cast<std::uint32_t>(
             reinterpret_cast<std::uint64_t>(&s_phase_done[0]) >> 2)) &
        4095u;
    // 定向集驱逐自旋 + 跳自槽。
    for (std::uint32_t t = 0; t < peCount; ++t) {
        if (t == tid) {
            continue;
        }
        std::uint32_t spins =
            0x9E3779B9u ^ (phase * 2654435761u) ^ (t * 0x85EBCA6Bu);
        while (s_phase_done[t] < phase) {
            // 寄存器驻留延迟链（零内存流量）+ 驱逐降频 1/64 轮。
            spins = spins * 2654435761u + 0x2545F491u;
            spins = spins * 2654435761u + 0x2545F492u;
            spins = spins * 2654435761u + 0x2545F493u;
            spins = spins * 2654435761u + 0x2545F494u;
            spins = spins * 2654435761u + 0x2545F495u;
            spins = spins * 2654435761u + 0x2545F496u;
            spins = spins * 2654435761u + 0x2545F497u;
            spins = spins * 2654435761u + 0x2545F498u;
            spins = spins * 2654435761u + 0x2545F499u;
            spins = spins * 2654435761u + 0x2545F49Au;
            spins = spins * 2654435761u + 0x2545F49Bu;
            spins = spins * 2654435761u + 0x2545F49Cu;
            spins = spins * 2654435761u + 0x2545F49Du;
            spins = spins * 2654435761u + 0x2545F49Eu;
            spins = spins * 2654435761u + 0x2545F49Fu;
            spins = spins * 2654435761u + 0x2545F4A0u;
            __asm__ volatile("" : "+r"(spins));
            if ((spins & 63u) == 0u) {
                for (std::uint32_t k = 1; k <= 5; ++k) {
                    (void)evict_base()[k * 16384u + word_off];
                }
            }
        }
    }
    compiler_barrier();
}

// ----------------------------------------------------------------------------
// evict_lines：对 [addr, addr+bytes) 按 64B 行做定向集驱逐。
//   每行取行基址的 set 字偏移，读驱逐区内 5 个同 set 别名行（16KB 步进），
//   依靠 set 关联冲突将目标行逐出 L1D（脏行则先写回 L2）。
// 用途：写方驱逐（标量写后跨 PE 可见）/ 读方驱逐（防陈旧 L1D 命中）。
// bytes 为 0 时无操作。对 TLSU(tile) 写路径，目标行须为写方独占
// （64B 行粒度，见 argmaxv2_rsplit.hpp 的 scratch 布局说明）。
// ----------------------------------------------------------------------------
inline void evict_lines(const volatile void *addr, std::size_t bytes) {
    compiler_barrier();
    const volatile char *base = static_cast<const volatile char *>(addr);
    for (std::size_t off = 0; off < bytes; off += 64) {
        const std::uint64_t line =
            (reinterpret_cast<std::uint64_t>(base + off)) &
            ~static_cast<std::uint64_t>(63);
        const std::uint32_t word_off =
            (static_cast<std::uint32_t>(line >> 2)) & 4095u;
        for (std::uint32_t k = 1; k <= 5; ++k) {
            (void)evict_base()[k * 16384u + word_off];
        }
    }
    compiler_barrier();
}

}  // namespace mt_sync
}  // namespace multi_thread
}  // namespace supernpu
