// TLSU 端到端用例的公共骨架。
//
// 判据不在 ELF 内：gfrun（黄金）与 gfsim 各跑一遍同一个 ELF，比对
// cross_model_result 区域的架构内存 dump。用例只负责产生激励并正常收尾。
// 比对由 SuperScalarModel 的 scripts/run_tlsu_compare.sh 驱动。
//
// 三条硬约束（都是实测踩出来的，见 docs/tlsu_e2e_stress_plan.md §3）：
//
//   1. 源数据必须烘进 ELF。标量访存目前没有 shared 路径 —— 标量 store 走私
//      有域（TLSU 前端 CommitStoreToL1 → L1D），而 TLOAD 读的是 shared/GM。
//      用标量循环填 src，那些字节到不了 tile load 能看见的地方。所以源一律
//      是 constinit 的全局量，由 ELF 加载器把 .data 内容直接灌进内存。
//
//   2. 收尾前必须留排空窗口。finisher 一旦退休就终止仿真，此前尚未全局可见
//      的写会从 dump 中丢失。
//
//   3. 不要用 M=1 的退化形状。编译器会发出 dstTile 为空的 TLOAD，gfrun 和
//      gfsim 都报 "TLOAD with empty dstTile!" 并 abort。
//
//   4. GM 行距一律按**逻辑元素数**计，不是字节。RowMajor 的第三个模板参数、
//      MakeTlsuPattern 的 STRIDE 都是元素数，B.IOR src1 编码的也是元素数
//      （PTO-TILE-MODEL-MEMORY-STRIDE）。TileOP-API 曾经在绑定处乘过
//      sizeof(DType)，Linx-TileOP-API@f35d3aa 已去掉；用例侧不需要、也不
//      允许再自己折算字节。
#ifndef TLSU_BENCH_HPP
#define TLSU_BENCH_HPP

#include <common/pto_tileop.hpp>
#include "benchmark.h"
#include "tlsu_finish.h"

using namespace pto;

// 自识别数据：每个元素编码自己的来源标记与行列号。
//
//   [31:28] tag   哪一份源图样
//   [27:16] r+1   行号，从 1 起
//   [15:0]  c+1   列号，从 1 起
//
// 行列都从 1 起，所以整个 32 位为 0 恒表示"从未被写"。hex dump 里一眼能看出
// 某一格来自哪份图样的哪一行哪一列 —— 地址错位、行距算错、beat 顺序颠倒、
// RAW/WAR 违例各有各的可见签名。
constexpr int TLSU_TAG_A = 0xA;   // 主图样
constexpr int TLSU_TAG_B = 0xB;   // 覆写图样（C 组的 "new"）
constexpr int TLSU_TAG_C = 0xC;   // 预置图样（C 组的 "pre"）
constexpr int TLSU_TAG_D = 0xD;   // witness 预填：区分"没写"和"写进了零"
constexpr int TLSU_TAG_PAD = 0xF; // 越界填充：出现在结果里即为过读

constexpr int TlsuTag(int tag, int r, int c)
{
    return (tag << 28) | ((r + 1) << 16) | (c + 1);
}

// 编号图样。R 组要的不是 4 份只读源，而是 16 份两两不同的 —— 只有 4 种内容时，
// "搬错了源"这类 bug 有 1/4 概率被同内容掩盖，判据看不见。
//
// 图样编号占用编码里空闲的 [27:24]：有效行 r+1 <= 8，即便 pad 行也只到 32，占到
// [21:16] 为止，不会与编号相碰。
//
//   [31:28] TLSU_TAG_ID  固定 1，表示"编号图样"
//   [27:24] pid          图样编号 0..15
//   [23:16] r+1          行号，从 1 起
//   [15:0]  c+1          列号，从 1 起
//
// hex dump 按 nibble 直读：0x15010001 = 编号图样 5、第 1 行、第 1 列。
constexpr int TLSU_TAG_ID = 0x1;

constexpr int TlsuIdTag(int pid, int r, int c)
{
    return (TLSU_TAG_ID << 28) | (pid << 24) | ((r + 1) << 16) | (c + 1);
}

// tile 的物理容量可能大于有效区。gfrun 的 TLOAD 行数上界是
//   totalRow = tileTotalSize / (totalCol * eleSize)      (TMAEngine.cpp:157)
// 由 tile 物理大小反推，validRow 只用于一个断言、不限制搬运范围。源对象只
// 按有效区分配就会被读越界。
//
// 于是源对象一律按 TLSU_PHYS_BYTES 分配（覆盖到目前见过的最大 size class），
// 有效区之外填 TLSU_TAG_PAD。过读因此是**可见的**而不是未定义行为：
// dump 里出现 0xF 开头的元素，就说明模型读出了有效区之外。
#define TLSU_PHYS_BYTES 16384

template <typename T, int VALID_ROWS, int COLS, int STRIDE = COLS>
struct TlsuPattern {
    static constexpr int kPhysRows = TLSU_PHYS_BYTES / (STRIDE * (int)sizeof(T));
    T v[kPhysRows * STRIDE];
};

// 行内 [0, COLS) 是有效列，[COLS, STRIDE) 是行间填充（strided 用例用）。
template <typename T, int VALID_ROWS, int COLS, int STRIDE = COLS>
static constexpr TlsuPattern<T, VALID_ROWS, COLS, STRIDE> MakeTlsuPattern(int tag)
{
    TlsuPattern<T, VALID_ROWS, COLS, STRIDE> d{};
    for (int r = 0; r < d.kPhysRows; ++r) {
        for (int c = 0; c < STRIDE; ++c) {
            bool valid = (r < VALID_ROWS) && (c < COLS);
            d.v[r * STRIDE + c] = (T)TlsuTag(valid ? tag : TLSU_TAG_PAD, r, c);
        }
    }
    return d;
}

// 与 MakeTlsuPattern 同构，只是有效区带图样编号。越界填充仍用 TLSU_TAG_PAD，所以
// "过读"与"搬错了源"在 dump 里是两种互不混淆的签名。
template <typename T, int VALID_ROWS, int COLS, int STRIDE = COLS>
static constexpr TlsuPattern<T, VALID_ROWS, COLS, STRIDE> MakeTlsuIdPattern(int pid)
{
    TlsuPattern<T, VALID_ROWS, COLS, STRIDE> d{};
    for (int r = 0; r < d.kPhysRows; ++r) {
        for (int c = 0; c < STRIDE; ++c) {
            bool valid = (r < VALID_ROWS) && (c < COLS);
            d.v[r * STRIDE + c] = (T)(valid ? TlsuIdTag(pid, r, c)
                                            : TlsuTag(TLSU_TAG_PAD, r, c));
        }
    }
    return d;
}

// 排空窗口。实测：把一笔标量 store 紧贴 tlsu_finish() 放置，它在 dump 里就
// 不见了；挪到 tile op 之前即正常。所有用例都必须在收尾前留出这段。
//
// 长度可由 -DTLSU_DRAIN=<n> 覆盖。这是个判别开关：某条写没进 dump 时，把它
// 调大若结果变了，问题在排空不够（用例的毛病）；不变则是模型真丢了写。
#ifndef TLSU_DRAIN
#define TLSU_DRAIN 20000
#endif

static inline void TlsuDrain()
{
    static volatile int spin;
    for (int k = 0; k < TLSU_DRAIN; ++k) {
        spin = spin + 1;
    }
}

// C 组把结果缓冲切成若干等长子区。默认按 TLSU_PHYS_BYTES 对齐，这样即使模型
// 按物理容量过写也不会串区；-DTLSU_REGION_STRIDE=<n> 可调小，用来判别"写没落
// 地"是否与子区间距（地址范围）有关。
#ifndef TLSU_REGION_STRIDE
#define TLSU_REGION_STRIDE TLSU_PHYS_BYTES
#endif

#define TLSU_REGION(i) ((int32_t *)(cross_model_result + (i) * TLSU_REGION_STRIDE))

#endif // TLSU_BENCH_HPP
