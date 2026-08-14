// R 组 —— 随机 TLOAD/TSTORE 序列。**本文件由 gen_random_case.py 生成，不要手改。**
//
//   种子   : 20260814
//   块数   : 200（91 个 TLOAD + 109 个 TSTORE）
//   形状   : 8x128 int32，稠密行距，基址 256 字节对齐
//   重新生成: python3 gen_random_case.py --seed 20260814 --blocks 200 \
//                 -o src/r1_random_seq_i32.cpp
//
// 随机的只有序列本身：块数、load/store 的交错、每一步用哪个 tile 寄存器、
// 读写哪一块内存。形状、dtype、行距全部固定 —— 失败时唯一的变量就是定序与
// 交错，不必再排除形状或寻址的嫌疑。
//
// 序列里天然会长出 RAW（store 后读同址）、WAR、WAW 和长距离 in-flight 交错，
// 这正是 S/C 两组用固定小序列够不到的地方：C 组每个用例只有两三个 tile op，
// 队列压力浅；这里一次 200 个块，LIQ/STQ/SCB 会被真正填满。
//
// 判据是 gfrun 的架构内存 dump（三方逐字节比对）。元素带自识别编码
// 0xT_rrr_cccc，dump 里能直接读出来源图样与行列号。
#include "tlsu_bench.hpp"

#define RESULT_SIZE (16 * 4096)
TLSU_RESULT_BUFFER(RESULT_SIZE);

constexpr int M = 8;
constexpr int N = 128;

alignas(256) constinit auto gSrc0 = MakeTlsuPattern<int32_t, M, N>(TLSU_TAG_A);
alignas(256) constinit auto gSrc1 = MakeTlsuPattern<int32_t, M, N>(TLSU_TAG_B);
alignas(256) constinit auto gSrc2 = MakeTlsuPattern<int32_t, M, N>(TLSU_TAG_C);
alignas(256) constinit auto gSrc3 = MakeTlsuPattern<int32_t, M, N>(TLSU_TAG_D);

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

    iter_t gs0(gSrc0.v);  auto s0 = gs0(0, 0);
    iter_t gs1(gSrc1.v);  auto s1 = gs1(0, 0);
    iter_t gs2(gSrc2.v);  auto s2 = gs2(0, 0);
    iter_t gs3(gSrc3.v);  auto s3 = gs3(0, 0);
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
    TLOAD(t5, s0);
    TSTORE(r13, t5);
    TLOAD(t1, r13);
    TLOAD(t3, r13);
    TSTORE(r6, t5);
    TSTORE(r1, t3);
    TLOAD(t5, r1);
    TSTORE(r2, t3);
    TSTORE(r3, t1);
    TSTORE(r3, t1);
    TSTORE(r14, t3);
    TSTORE(r8, t3);
    TSTORE(r14, t5);
    TSTORE(r2, t1);
    TLOAD(t1, r14);
    TSTORE(r13, t3);
    TLOAD(t7, r14);
    TLOAD(t6, r1);
    TLOAD(t4, s2);
    TLOAD(t5, s1);
    TLOAD(t3, s3);
    TSTORE(r9, t3);
    TSTORE(r2, t6);
    TSTORE(r6, t5);
    TLOAD(t0, s1);
    TLOAD(t4, r3);
    TLOAD(t7, s0);
    TLOAD(t0, r8);
    TSTORE(r6, t3);
    TSTORE(r15, t0);
    TLOAD(t2, s0);
    TLOAD(t6, s1);
    TSTORE(r10, t3);
    TLOAD(t4, r6);
    TLOAD(t0, r10);
    TSTORE(r13, t2);
    TLOAD(t4, s1);
    TLOAD(t5, s2);
    TLOAD(t0, s1);
    TLOAD(t5, r15);
    TSTORE(r8, t1);
    TLOAD(t6, r14);
    TLOAD(t2, r13);
    TSTORE(r6, t2);
    TLOAD(t6, s2);
    TSTORE(r4, t1);
    TSTORE(r11, t7);
    TSTORE(r8, t1);
    TLOAD(t5, r2);
    TLOAD(t5, s3);
    TSTORE(r10, t7);
    TLOAD(t7, s2);
    TLOAD(t2, s1);
    TSTORE(r8, t6);
    TSTORE(r2, t5);
    TSTORE(r0, t7);
    TSTORE(r5, t3);
    TSTORE(r6, t2);
    TLOAD(t7, r10);
    TLOAD(t6, s0);
    TLOAD(t4, s1);
    TSTORE(r13, t4);
    TLOAD(t7, r4);
    TSTORE(r10, t3);
    TSTORE(r11, t2);
    TSTORE(r13, t1);
    TSTORE(r13, t2);
    TLOAD(t5, s0);
    TSTORE(r3, t7);
    TSTORE(r3, t5);
    TSTORE(r1, t2);
    TSTORE(r10, t7);
    TSTORE(r13, t4);
    TLOAD(t2, r5);
    TSTORE(r11, t2);
    TLOAD(t4, r6);
    TLOAD(t1, r14);
    TLOAD(t3, r14);
    TSTORE(r13, t0);
    TSTORE(r3, t3);
    TSTORE(r8, t4);
    TLOAD(t4, s2);
    TLOAD(t6, r1);
    TSTORE(r11, t0);
    TLOAD(t0, r2);
    TLOAD(t6, r15);
    TLOAD(t6, s0);
    TSTORE(r6, t4);
    TSTORE(r12, t4);
    TSTORE(r7, t3);
    TLOAD(t4, s3);
    TSTORE(r14, t0);
    TLOAD(t2, r14);
    TSTORE(r10, t2);
    TLOAD(t3, r7);
    TLOAD(t4, r13);
    TLOAD(t1, s1);
    TLOAD(t4, s1);
    TSTORE(r8, t6);
    TLOAD(t1, s0);
    TLOAD(t4, r6);
    TLOAD(t7, r0);
    TSTORE(r15, t2);
    TLOAD(t1, s1);
    TSTORE(r8, t5);
    TSTORE(r1, t3);
    TSTORE(r15, t4);
    TLOAD(t7, s2);
    TSTORE(r0, t3);
    TSTORE(r3, t6);
    TSTORE(r9, t2);
    TSTORE(r6, t6);
    TSTORE(r4, t5);
    TSTORE(r0, t7);
    TLOAD(t4, s0);
    TSTORE(r6, t4);
    TLOAD(t6, r3);
    TLOAD(t2, s2);
    TLOAD(t4, r9);
    TLOAD(t0, r2);
    TSTORE(r2, t2);
    TSTORE(r11, t1);
    TLOAD(t5, s0);
    TSTORE(r13, t7);
    TSTORE(r15, t4);
    TSTORE(r11, t0);
    TSTORE(r1, t2);
    TSTORE(r5, t7);
    TLOAD(t1, s2);
    TLOAD(t0, r2);
    TSTORE(r6, t3);
    TLOAD(t2, r15);
    TSTORE(r15, t6);
    TLOAD(t6, r8);
    TLOAD(t2, r12);
    TSTORE(r1, t6);
    TSTORE(r5, t2);
    TSTORE(r2, t5);
    TSTORE(r14, t6);
    TSTORE(r6, t5);
    TSTORE(r11, t7);
    TSTORE(r6, t6);
    TSTORE(r7, t6);
    TLOAD(t7, r7);
    TLOAD(t7, s3);
    TLOAD(t0, r1);
    TLOAD(t7, s1);
    TSTORE(r7, t3);
    TSTORE(r10, t0);
    TLOAD(t6, s3);
    TLOAD(t6, s1);
    TLOAD(t5, r11);
    TSTORE(r9, t4);
    TLOAD(t7, s2);
    TLOAD(t2, s0);
    TSTORE(r8, t1);
    TSTORE(r4, t7);
    TLOAD(t5, s1);
    TLOAD(t1, r7);
    TLOAD(t6, r1);
    TSTORE(r10, t2);
    TSTORE(r8, t1);
    TLOAD(t1, s3);
    TSTORE(r3, t6);
    TSTORE(r10, t3);
    TSTORE(r6, t1);
    TSTORE(r4, t7);
    TSTORE(r0, t2);
    TSTORE(r5, t6);
    TSTORE(r12, t6);
    TLOAD(t7, r3);
    TSTORE(r14, t0);
    TLOAD(t4, s2);
    TLOAD(t1, r11);
    TLOAD(t4, r8);
    TLOAD(t1, s3);
    TSTORE(r14, t0);
    TLOAD(t3, s3);
    TSTORE(r11, t3);
    TLOAD(t2, s0);
    TSTORE(r0, t0);
    TSTORE(r0, t2);
    TLOAD(t4, s1);
    TSTORE(r1, t0);
    TSTORE(r2, t2);
    TSTORE(r5, t5);
    TLOAD(t3, r11);
    TSTORE(r3, t0);
    TSTORE(r3, t3);
    TLOAD(t0, s3);
    TSTORE(r8, t1);
    TSTORE(r4, t5);
    TLOAD(t5, s0);
    TSTORE(r7, t5);
    TSTORE(r0, t0);
    TLOAD(t3, r10);
    TSTORE(r9, t6);
    TLOAD(t7, s1);
    TSTORE(r0, t1);
    TSTORE(r12, t7);
    BENCHEND;

    TlsuDrain();
    tlsu_finish(1);
    return 0;
}
