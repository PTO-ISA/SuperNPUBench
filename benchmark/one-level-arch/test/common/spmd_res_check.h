#ifndef SPMD_RES_CHECK_H
#define SPMD_RES_CHECK_H

// 4-PE SPMD res_check 收尾协议 helper（issue #489）。
//
// 背景：gfrun hosted SMT4 下 runtime（group_worker_runtime.c）把 PE1..PE3 的 PC
// 指到 __linx_group_worker_start，它调 main() 后 park 死循环、绝不 exit_group，由
// PE0 的 exit_group 统一收尾。但 4 个 PE 仍各自跑完整 main()——原始 driver 让 4 个
// PE 都 writeBinaryFile(整个 buffer, O_TRUNC)，互相截断覆盖，且谁先写取决于时序：
// 小 shape 只留下 PE0/末段碎片，大 shape 碰巧全过（issue #489 官方定性）。
//
// 修复：把文件 I/O 收敛到 leader(tid 0)。读侧用「worker 等 leader 读完」的输入屏障
// （对照 matmul_shared.cpp 已验证的 leader_ready idiom）；写侧仅 leader 落盘。
//
// 为何写侧不需要（也不能用）「leader 等 worker」的输出屏障：
//   1) 本模型无硬件 barrier 原语（TileOP API 仅暴露 get_thread_idx），唯一同步手段
//      是共享内存自旋。实测在 gfrun 下「leader 忙等 worker done」方向的自旋会 hang
//      （worker 等 leader 方向 OK）——与 issue 文档「core-level barrier 需 gfrun 支持」
//      一致，故不可用。
//   2) 不需要：kernel 把工作按 tid **连续切分**，leader(tid 0) 恒拿 **ceiling 份额**
//      （nontail: SubKb = numKb/kPeNum + (tid<rem)；tail: SubM 同式），即 leader 的
//      计算量 ≥ 任何 worker。配合 gfrun execWidth=1 的**块级锁步**（每步每线程各推进
//      1 个 block）+ leader 额外承担文件 I/O，PE0 **必最后完成**——它落盘时所有 worker
//      的分片早已写入**共享** buffer（y/scale 为 static 全局，4 PE 同地址空间；实测
//      跨 PE 写可见）。故 PE0-only 写盘的完整性由**构造保证**，而非时序运气。
//      端到端验证：Axis=512/Post=256（numKb=16，4 PE 全活）output=pass、scale=pass
//      MaxAE=0，4 段各 32768B 全填齐。
//
// 仅在 RES_CHECK 精度流程中使用（此时 ENABLE_BINARY_OUTPUT 亦已定义，read/write
// Binary 为真实实现）。ready 旗标用函数内 static（单 TU 单实例、落 .bss 由 4 PE 共享、
// 零初始化免 C++ static-init guard）；__asm__ volatile("":::"memory") 为编译器屏障。
//
// 注（与本 helper 无关的 gfrun 既有限制）：部分 shape 在 gfrun res_check 下不可跑——
// Post=64 等窄自由轴的 4-PE res_check 会 hang（实测：**原始全写代码亦 hang**，非本
// 收尾协议引入，属 gfrun 多 PE 文件 I/O 基础设施缺陷）；Axis=128 触发 decode 断言
// （单 PE 亦崩）。验证请选可跑 shape（如 Axis=512/Post=256）。

#include <cstddef>
#include <cstdint>

#include <common/pto_tileop.hpp>  // get_thread_idx() -> uint32_t (0..3)
#include "fileop.h"               // readBinaryFile / writeBinaryFile

template <int kPeNum = 4>
struct SpmdResCheck {
    uint32_t tid;
    SpmdResCheck() : tid(get_thread_idx()) {}

    // 所有 PE 调用：leader(tid 0) 读文件，worker 自旋等 leader 读完（输入屏障）。
    void leader_load(const char *path, void *buf, size_t bytes) {
        static volatile int ready = 0;  // 零初始化 → .bss，4 PE 共享单实例
        if (tid == 0) {
            readBinaryFile(path, reinterpret_cast<uint8_t *>(buf), bytes);
            __asm__ volatile("" : : : "memory");
            ready = 1;
        } else {
            while (!ready) {
            }
            __asm__ volatile("" : : : "memory");
        }
    }

    // 仅 leader(tid 0) 落盘；worker 返回 false → main 返回 → runtime park（不 exit_group）。
    // 完整性由 leader-heaviest 锁步保证（见文件头说明），无需输出屏障。
    bool leader_should_write() const { return tid == 0; }
};

#endif  // SPMD_RES_CHECK_H
