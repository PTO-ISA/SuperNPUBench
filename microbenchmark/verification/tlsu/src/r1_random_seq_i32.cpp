// R 组 —— 随机 TLOAD/TSTORE 序列。**本文件由 gen_random_case.py 生成，不要手改。**
//
//   种子   : 20260818
//   初始化 : 32 块（16 个 TLOAD + 16 个 TSTORE），每个区装一份唯一图样
//   随机体 : 200 块（103 个 TLOAD + 97 个 TSTORE），只在区与区之间搬运
//   形状   : 8x128 int32，稠密行距，基址 256 字节对齐
//   重新生成: python3 gen_random_case.py --seed 20260818 --blocks 200 \
//                 -o src/r1_random_seq_i32.cpp
//
// 随机的只有序列本身：load/store 的交错、每一步用哪个 tile 寄存器、读写哪一块
// 内存。形状、dtype、行距全部固定 —— 失败时唯一的变量就是定序与交错，不必再
// 排除形状或寻址的嫌疑。
//
// 随机体不读只读图样：初始化之后 16 个区全部有内容，之后每一次 TLOAD 读的
// 都是可能刚被写过的区。序列里因此密集地长出 RAW（store 后读同址）、WAR、WAW
// 和长距离 in-flight 交错 —— 这正是 S/C 两组用固定小序列够不到的地方：C 组每个
// 用例只有两三个 tile op，队列压力浅；这里一次 200 个块，LIQ/STQ/SCB 会被真正
// 填满。
//
// 每个区的初始图样互不相同（区号在元素值的 [27:24]），所以"错读了别的区"不会被
// 同内容掩盖 —— 任何一次搬错源都在 dump 里可见。元素编码见 tlsu_bench.hpp：
//   0xT_R_rr_cccc = 图样 tag、区号、行号、列号，hex dump 按 nibble 直读。
#include "tlsu_bench.hpp"

#define RESULT_SIZE (16 * 4096)
TLSU_RESULT_BUFFER(RESULT_SIZE);

constexpr int M = 8;
constexpr int N = 128;

// 每个区一份唯一的初始图样：区号编进元素值，错读别的区在 dump 里可见。
alignas(256) constinit auto gWs0 = MakeTlsuWsPattern<int32_t, M, N>(0);
alignas(256) constinit auto gWs1 = MakeTlsuWsPattern<int32_t, M, N>(1);
alignas(256) constinit auto gWs2 = MakeTlsuWsPattern<int32_t, M, N>(2);
alignas(256) constinit auto gWs3 = MakeTlsuWsPattern<int32_t, M, N>(3);
alignas(256) constinit auto gWs4 = MakeTlsuWsPattern<int32_t, M, N>(4);
alignas(256) constinit auto gWs5 = MakeTlsuWsPattern<int32_t, M, N>(5);
alignas(256) constinit auto gWs6 = MakeTlsuWsPattern<int32_t, M, N>(6);
alignas(256) constinit auto gWs7 = MakeTlsuWsPattern<int32_t, M, N>(7);
alignas(256) constinit auto gWs8 = MakeTlsuWsPattern<int32_t, M, N>(8);
alignas(256) constinit auto gWs9 = MakeTlsuWsPattern<int32_t, M, N>(9);
alignas(256) constinit auto gWs10 = MakeTlsuWsPattern<int32_t, M, N>(10);
alignas(256) constinit auto gWs11 = MakeTlsuWsPattern<int32_t, M, N>(11);
alignas(256) constinit auto gWs12 = MakeTlsuWsPattern<int32_t, M, N>(12);
alignas(256) constinit auto gWs13 = MakeTlsuWsPattern<int32_t, M, N>(13);
alignas(256) constinit auto gWs14 = MakeTlsuWsPattern<int32_t, M, N>(14);
alignas(256) constinit auto gWs15 = MakeTlsuWsPattern<int32_t, M, N>(15);

int main()
{
    using gm_t = global_tensor<int32_t, RowMajor<M, N>>;
    using tile_t = Tile<Location::Vec, int32_t, M, N, BLayout::RowMajor>;
    using iter_t = global_iterator<gm_t, tile_t>;

    // 区域基址按 tile 紧排；TILE_BYTES 是 256 的整数倍，
    // 所以每个区域的基址天然 256 字节对齐。
    int32_t *R0 = (int32_t *)(cross_model_result + 0 * 4096);
    int32_t *R1 = (int32_t *)(cross_model_result + 1 * 4096);
    int32_t *R2 = (int32_t *)(cross_model_result + 2 * 4096);
    int32_t *R3 = (int32_t *)(cross_model_result + 3 * 4096);
    int32_t *R4 = (int32_t *)(cross_model_result + 4 * 4096);
    int32_t *R5 = (int32_t *)(cross_model_result + 5 * 4096);
    int32_t *R6 = (int32_t *)(cross_model_result + 6 * 4096);
    int32_t *R7 = (int32_t *)(cross_model_result + 7 * 4096);
    int32_t *R8 = (int32_t *)(cross_model_result + 8 * 4096);
    int32_t *R9 = (int32_t *)(cross_model_result + 9 * 4096);
    int32_t *R10 = (int32_t *)(cross_model_result + 10 * 4096);
    int32_t *R11 = (int32_t *)(cross_model_result + 11 * 4096);
    int32_t *R12 = (int32_t *)(cross_model_result + 12 * 4096);
    int32_t *R13 = (int32_t *)(cross_model_result + 13 * 4096);
    int32_t *R14 = (int32_t *)(cross_model_result + 14 * 4096);
    int32_t *R15 = (int32_t *)(cross_model_result + 15 * 4096);

    iter_t gw0(gWs0.v);  auto w0 = gw0(0, 0);
    iter_t gw1(gWs1.v);  auto w1 = gw1(0, 0);
    iter_t gw2(gWs2.v);  auto w2 = gw2(0, 0);
    iter_t gw3(gWs3.v);  auto w3 = gw3(0, 0);
    iter_t gw4(gWs4.v);  auto w4 = gw4(0, 0);
    iter_t gw5(gWs5.v);  auto w5 = gw5(0, 0);
    iter_t gw6(gWs6.v);  auto w6 = gw6(0, 0);
    iter_t gw7(gWs7.v);  auto w7 = gw7(0, 0);
    iter_t gw8(gWs8.v);  auto w8 = gw8(0, 0);
    iter_t gw9(gWs9.v);  auto w9 = gw9(0, 0);
    iter_t gw10(gWs10.v);  auto w10 = gw10(0, 0);
    iter_t gw11(gWs11.v);  auto w11 = gw11(0, 0);
    iter_t gw12(gWs12.v);  auto w12 = gw12(0, 0);
    iter_t gw13(gWs13.v);  auto w13 = gw13(0, 0);
    iter_t gw14(gWs14.v);  auto w14 = gw14(0, 0);
    iter_t gw15(gWs15.v);  auto w15 = gw15(0, 0);
    iter_t gr0(R0);  auto r0 = gr0(0, 0);
    iter_t gr1(R1);  auto r1 = gr1(0, 0);
    iter_t gr2(R2);  auto r2 = gr2(0, 0);
    iter_t gr3(R3);  auto r3 = gr3(0, 0);
    iter_t gr4(R4);  auto r4 = gr4(0, 0);
    iter_t gr5(R5);  auto r5 = gr5(0, 0);
    iter_t gr6(R6);  auto r6 = gr6(0, 0);
    iter_t gr7(R7);  auto r7 = gr7(0, 0);
    iter_t gr8(R8);  auto r8 = gr8(0, 0);
    iter_t gr9(R9);  auto r9 = gr9(0, 0);
    iter_t gr10(R10);  auto r10 = gr10(0, 0);
    iter_t gr11(R11);  auto r11 = gr11(0, 0);
    iter_t gr12(R12);  auto r12 = gr12(0, 0);
    iter_t gr13(R13);  auto r13 = gr13(0, 0);
    iter_t gr14(R14);  auto r14 = gr14(0, 0);
    iter_t gr15(R15);  auto r15 = gr15(0, 0);

    tile_t t0, t1, t2, t3, t4, t5, t6, t7;

    BENCHSTART;
    // ── 初始化：每个区装入自己的那份图样 ──
    TLOAD(t0, w0);
    TSTORE(r0, t0);
    TLOAD(t1, w1);
    TSTORE(r1, t1);
    TLOAD(t2, w2);
    TSTORE(r2, t2);
    TLOAD(t3, w3);
    TSTORE(r3, t3);
    TLOAD(t4, w4);
    TSTORE(r4, t4);
    TLOAD(t5, w5);
    TSTORE(r5, t5);
    TLOAD(t6, w6);
    TSTORE(r6, t6);
    TLOAD(t7, w7);
    TSTORE(r7, t7);
    TLOAD(t0, w8);
    TSTORE(r8, t0);
    TLOAD(t1, w9);
    TSTORE(r9, t1);
    TLOAD(t2, w10);
    TSTORE(r10, t2);
    TLOAD(t3, w11);
    TSTORE(r11, t3);
    TLOAD(t4, w12);
    TSTORE(r12, t4);
    TLOAD(t5, w13);
    TSTORE(r13, t5);
    TLOAD(t6, w14);
    TSTORE(r14, t6);
    TLOAD(t7, w15);
    TSTORE(r15, t7);

    // ── 随机体：只在区与区之间搬运 ──
    TSTORE(r7, t7);
    TSTORE(r15, t4);
    TLOAD(t5, r7);
    TLOAD(t3, r3);
    TLOAD(t4, r14);
    TLOAD(t1, r5);
    TLOAD(t3, r14);
    TLOAD(t3, r13);
    TSTORE(r7, t0);
    TLOAD(t0, r7);
    TSTORE(r10, t0);
    TSTORE(r1, t4);
    TLOAD(t1, r7);
    TSTORE(r7, t7);
    TLOAD(t3, r12);
    TLOAD(t0, r11);
    TLOAD(t1, r8);
    TLOAD(t5, r14);
    TLOAD(t6, r0);
    TLOAD(t3, r11);
    TSTORE(r0, t3);
    TLOAD(t6, r4);
    TSTORE(r3, t6);
    TSTORE(r6, t4);
    TLOAD(t7, r14);
    TSTORE(r4, t2);
    TLOAD(t2, r10);
    TLOAD(t5, r13);
    TLOAD(t2, r12);
    TLOAD(t3, r5);
    TSTORE(r14, t6);
    TSTORE(r0, t6);
    TLOAD(t2, r7);
    TLOAD(t1, r14);
    TSTORE(r12, t4);
    TLOAD(t5, r11);
    TLOAD(t2, r7);
    TSTORE(r6, t6);
    TLOAD(t7, r3);
    TSTORE(r5, t5);
    TLOAD(t4, r0);
    TSTORE(r15, t0);
    TLOAD(t5, r10);
    TSTORE(r6, t3);
    TLOAD(t1, r14);
    TLOAD(t6, r8);
    TSTORE(r13, t2);
    TLOAD(t1, r9);
    TSTORE(r12, t6);
    TLOAD(t5, r12);
    TSTORE(r2, t3);
    TLOAD(t6, r13);
    TSTORE(r4, t6);
    TSTORE(r1, t4);
    TSTORE(r1, t4);
    TSTORE(r8, t4);
    TLOAD(t0, r1);
    TLOAD(t7, r14);
    TLOAD(t4, r5);
    TSTORE(r11, t1);
    TLOAD(t5, r5);
    TSTORE(r9, t0);
    TLOAD(t6, r14);
    TSTORE(r15, t3);
    TSTORE(r8, t4);
    TLOAD(t6, r9);
    TLOAD(t7, r11);
    TSTORE(r5, t2);
    TLOAD(t4, r12);
    TSTORE(r5, t6);
    TLOAD(t2, r11);
    TLOAD(t5, r8);
    TLOAD(t7, r2);
    TLOAD(t2, r3);
    TSTORE(r2, t3);
    TSTORE(r10, t5);
    TLOAD(t6, r0);
    TSTORE(r12, t6);
    TSTORE(r1, t4);
    TSTORE(r14, t3);
    TSTORE(r1, t1);
    TSTORE(r2, t5);
    TLOAD(t4, r3);
    TLOAD(t3, r4);
    TSTORE(r12, t3);
    TLOAD(t6, r14);
    TLOAD(t6, r12);
    TSTORE(r8, t5);
    TSTORE(r2, t4);
    TLOAD(t1, r11);
    TLOAD(t7, r7);
    TSTORE(r2, t3);
    TSTORE(r0, t4);
    TSTORE(r5, t5);
    TLOAD(t7, r1);
    TSTORE(r13, t0);
    TSTORE(r2, t2);
    TLOAD(t5, r10);
    TSTORE(r4, t3);
    TSTORE(r14, t0);
    TLOAD(t7, r10);
    TSTORE(r5, t7);
    TSTORE(r5, t6);
    TSTORE(r9, t1);
    TSTORE(r3, t1);
    TLOAD(t4, r2);
    TLOAD(t1, r0);
    TSTORE(r1, t1);
    TSTORE(r11, t5);
    TLOAD(t6, r13);
    TLOAD(t6, r0);
    TSTORE(r13, t2);
    TLOAD(t7, r9);
    TSTORE(r6, t0);
    TSTORE(r11, t2);
    TLOAD(t0, r10);
    TSTORE(r12, t2);
    TSTORE(r5, t4);
    TLOAD(t6, r8);
    TSTORE(r13, t6);
    TLOAD(t2, r11);
    TLOAD(t7, r15);
    TLOAD(t7, r4);
    TLOAD(t1, r3);
    TSTORE(r4, t0);
    TLOAD(t2, r7);
    TLOAD(t2, r7);
    TLOAD(t2, r0);
    TLOAD(t0, r13);
    TSTORE(r11, t3);
    TLOAD(t6, r13);
    TLOAD(t5, r2);
    TSTORE(r6, t4);
    TLOAD(t4, r7);
    TLOAD(t2, r6);
    TLOAD(t2, r7);
    TLOAD(t5, r3);
    TLOAD(t1, r13);
    TSTORE(r14, t4);
    TLOAD(t5, r2);
    TSTORE(r6, t0);
    TLOAD(t4, r14);
    TLOAD(t4, r9);
    TSTORE(r4, t5);
    TLOAD(t0, r7);
    TSTORE(r7, t3);
    TSTORE(r9, t1);
    TSTORE(r3, t3);
    TLOAD(t4, r15);
    TLOAD(t0, r9);
    TSTORE(r9, t4);
    TSTORE(r11, t0);
    TLOAD(t5, r5);
    TSTORE(r0, t6);
    TSTORE(r9, t4);
    TLOAD(t3, r14);
    TLOAD(t3, r7);
    TLOAD(t1, r0);
    TSTORE(r12, t1);
    TSTORE(r3, t6);
    TSTORE(r15, t3);
    TSTORE(r0, t2);
    TLOAD(t6, r2);
    TSTORE(r14, t0);
    TLOAD(t0, r0);
    TSTORE(r15, t5);
    TSTORE(r13, t4);
    TLOAD(t4, r14);
    TSTORE(r10, t3);
    TLOAD(t4, r10);
    TLOAD(t2, r13);
    TLOAD(t7, r4);
    TSTORE(r13, t1);
    TSTORE(r3, t5);
    TSTORE(r7, t4);
    TLOAD(t3, r7);
    TLOAD(t4, r5);
    TSTORE(r6, t1);
    TSTORE(r10, t4);
    TLOAD(t6, r0);
    TLOAD(t4, r0);
    TLOAD(t4, r15);
    TLOAD(t7, r0);
    TSTORE(r12, t2);
    TLOAD(t2, r8);
    TLOAD(t4, r6);
    TSTORE(r8, t2);
    TLOAD(t7, r5);
    TSTORE(r0, t1);
    TSTORE(r1, t0);
    TSTORE(r5, t0);
    TSTORE(r4, t1);
    TSTORE(r7, t3);
    TSTORE(r13, t3);
    TSTORE(r7, t3);
    TSTORE(r4, t5);
    TSTORE(r9, t2);
    TLOAD(t3, r13);
    TLOAD(t7, r9);
    TSTORE(r12, t7);
    BENCHEND;

    TlsuDrain();
    tlsu_finish(1);
    return 0;
}
