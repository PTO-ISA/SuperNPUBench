// R 组 —— 随机 TLOAD/TSTORE 序列。**本文件由 gen_random_case.py 生成，不要手改。**
//
//   种子   : 20260818
//   块数   : 200（99 个 TLOAD + 101 个 TSTORE）
//   只读源 : 16 份，两两不同（编号在元素值的 [27:24]）
//   形状   : 8x128 int32，稠密行距，基址 256 字节对齐
//   重新生成: python3 gen_random_case.py --seed 20260818 --blocks 200 \
//                 -o src/r1_random_seq_i32.cpp
//
// 随机的只有序列本身：块数、load/store 的交错、每一步用哪个 tile 寄存器、读写
// 哪一块内存。形状、dtype、行距全部固定 —— 失败时唯一的变量就是定序与交错，
// 不必再排除形状或寻址的嫌疑。
//
// 序列里天然会长出 RAW（store 后读同址）、WAR、WAW 和长距离 in-flight 交错，
// 这正是 S/C 两组用固定小序列够不到的地方：C 组每个用例只有两三个 tile op，
// 队列压力浅；这里一次 200 个块，LIQ/STQ/SCB 会被真正填满。
//
// 16 份只读图样两两不同，所以"搬错了源"不会被同内容掩盖，任何一次都在 dump 里
// 可见；而 TSTORE 只复制不创造，靠这些 src 装载持续注入新编号，序列再长也不会
// 塌缩成少数几种内容。元素编码见 tlsu_bench.hpp：
//   0x1_P_rr_cccc = 编号图样 P、第 rr 行、第 cccc 列，hex dump 按 nibble 直读。
#include "tlsu_bench.hpp"

#define RESULT_SIZE (16 * 4096)
TLSU_RESULT_BUFFER(RESULT_SIZE);

constexpr int M = 8;
constexpr int N = 128;

// 16 份两两不同的只读图样：编号编进元素值，搬错源在 dump 里可见。
alignas(256) constinit auto gSrc0 = MakeTlsuIdPattern<int32_t, M, N>(0);
alignas(256) constinit auto gSrc1 = MakeTlsuIdPattern<int32_t, M, N>(1);
alignas(256) constinit auto gSrc2 = MakeTlsuIdPattern<int32_t, M, N>(2);
alignas(256) constinit auto gSrc3 = MakeTlsuIdPattern<int32_t, M, N>(3);
alignas(256) constinit auto gSrc4 = MakeTlsuIdPattern<int32_t, M, N>(4);
alignas(256) constinit auto gSrc5 = MakeTlsuIdPattern<int32_t, M, N>(5);
alignas(256) constinit auto gSrc6 = MakeTlsuIdPattern<int32_t, M, N>(6);
alignas(256) constinit auto gSrc7 = MakeTlsuIdPattern<int32_t, M, N>(7);
alignas(256) constinit auto gSrc8 = MakeTlsuIdPattern<int32_t, M, N>(8);
alignas(256) constinit auto gSrc9 = MakeTlsuIdPattern<int32_t, M, N>(9);
alignas(256) constinit auto gSrc10 = MakeTlsuIdPattern<int32_t, M, N>(10);
alignas(256) constinit auto gSrc11 = MakeTlsuIdPattern<int32_t, M, N>(11);
alignas(256) constinit auto gSrc12 = MakeTlsuIdPattern<int32_t, M, N>(12);
alignas(256) constinit auto gSrc13 = MakeTlsuIdPattern<int32_t, M, N>(13);
alignas(256) constinit auto gSrc14 = MakeTlsuIdPattern<int32_t, M, N>(14);
alignas(256) constinit auto gSrc15 = MakeTlsuIdPattern<int32_t, M, N>(15);

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
    iter_t gs4(gSrc4.v);  auto s4 = gs4(0, 0);
    iter_t gs5(gSrc5.v);  auto s5 = gs5(0, 0);
    iter_t gs6(gSrc6.v);  auto s6 = gs6(0, 0);
    iter_t gs7(gSrc7.v);  auto s7 = gs7(0, 0);
    iter_t gs8(gSrc8.v);  auto s8 = gs8(0, 0);
    iter_t gs9(gSrc9.v);  auto s9 = gs9(0, 0);
    iter_t gs10(gSrc10.v);  auto s10 = gs10(0, 0);
    iter_t gs11(gSrc11.v);  auto s11 = gs11(0, 0);
    iter_t gs12(gSrc12.v);  auto s12 = gs12(0, 0);
    iter_t gs13(gSrc13.v);  auto s13 = gs13(0, 0);
    iter_t gs14(gSrc14.v);  auto s14 = gs14(0, 0);
    iter_t gs15(gSrc15.v);  auto s15 = gs15(0, 0);
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
    TLOAD(t5, s3);
    TLOAD(t3, s6);
    TLOAD(t4, s15);
    TLOAD(t5, s7);
    TLOAD(t3, s3);
    TLOAD(t4, s14);
    TLOAD(t1, s5);
    TLOAD(t3, s14);
    TLOAD(t3, s13);
    TSTORE(r7, t1);
    TLOAD(t0, r7);
    TLOAD(t5, r7);
    TLOAD(t0, s3);
    TLOAD(t3, r7);
    TLOAD(t3, s12);
    TLOAD(t0, r7);
    TSTORE(r10, t3);
    TLOAD(t6, s0);
    TLOAD(t3, r7);
    TSTORE(r13, t0);
    TLOAD(t2, r13);
    TSTORE(r14, t2);
    TSTORE(r6, t4);
    TLOAD(t7, r10);
    TSTORE(r10, t2);
    TLOAD(t5, s13);
    TLOAD(t2, r7);
    TLOAD(t2, r13);
    TLOAD(t7, r6);
    TLOAD(t2, r14);
    TLOAD(t1, r10);
    TLOAD(t4, s12);
    TLOAD(t5, r7);
    TLOAD(t2, r13);
    TSTORE(r14, t3);
    TSTORE(r11, t7);
    TSTORE(r0, t4);
    TSTORE(r15, t0);
    TLOAD(t5, r10);
    TLOAD(t3, s2);
    TSTORE(r13, t3);
    TLOAD(t1, r7);
    TSTORE(r3, t2);
    TLOAD(t4, r13);
    TLOAD(t6, s11);
    TLOAD(t6, r7);
    TSTORE(r12, t1);
    TLOAD(t2, s12);
    TSTORE(r9, t0);
    TSTORE(r9, t3);
    TSTORE(r9, t0);
    TSTORE(r0, t2);
    TLOAD(t0, s15);
    TLOAD(t2, s5);
    TSTORE(r11, t1);
    TLOAD(t5, s5);
    TSTORE(r9, t0);
    TLOAD(t6, r6);
    TSTORE(r0, t1);
    TSTORE(r12, t1);
    TSTORE(r14, t0);
    TLOAD(t1, r6);
    TSTORE(r12, t4);
    TSTORE(r5, t6);
    TLOAD(t2, r11);
    TSTORE(r14, t4);
    TSTORE(r4, t3);
    TSTORE(r2, t3);
    TSTORE(r10, t5);
    TLOAD(t6, r3);
    TSTORE(r1, t6);
    TSTORE(r11, t0);
    TSTORE(r14, t3);
    TSTORE(r1, t1);
    TSTORE(r2, t5);
    TLOAD(t4, s6);
    TSTORE(r6, t6);
    TLOAD(t6, s12);
    TSTORE(r13, t7);
    TSTORE(r11, t1);
    TSTORE(r2, t4);
    TLOAD(t1, s15);
    TSTORE(r6, t7);
    TSTORE(r0, t4);
    TSTORE(r5, t5);
    TLOAD(t7, s4);
    TSTORE(r0, t6);
    TSTORE(r2, t2);
    TLOAD(t5, s10);
    TSTORE(r4, t3);
    TSTORE(r14, t0);
    TLOAD(t7, s10);
    TSTORE(r5, t7);
    TSTORE(r5, t6);
    TSTORE(r9, t1);
    TSTORE(r3, t1);
    TLOAD(t4, r0);
    TSTORE(r14, t0);
    TSTORE(r1, t1);
    TSTORE(r11, t5);
    TLOAD(t6, s12);
    TSTORE(r13, t2);
    TLOAD(t7, s5);
    TSTORE(r6, t0);
    TSTORE(r11, t2);
    TLOAD(t0, s10);
    TSTORE(r12, t2);
    TSTORE(r5, t4);
    TLOAD(t6, s11);
    TLOAD(t6, s4);
    TLOAD(t7, r6);
    TSTORE(r2, t7);
    TSTORE(r0, t3);
    TLOAD(t2, s4);
    TSTORE(r5, t7);
    TLOAD(t2, r10);
    TSTORE(r6, t7);
    TSTORE(r13, t0);
    TSTORE(r2, t5);
    TSTORE(r6, t4);
    TLOAD(t4, s5);
    TSTORE(r4, t7);
    TSTORE(r3, t5);
    TLOAD(t1, s13);
    TSTORE(r14, t4);
    TLOAD(t5, r12);
    TSTORE(r9, t3);
    TLOAD(t7, s8);
    TSTORE(r11, t5);
    TLOAD(t0, s8);
    TLOAD(t3, r1);
    TLOAD(t4, s3);
    TLOAD(t4, s1);
    TLOAD(t3, r4);
    TSTORE(r11, t0);
    TLOAD(t5, s5);
    TSTORE(r0, t6);
    TSTORE(r9, t4);
    TLOAD(t3, r1);
    TLOAD(t3, s8);
    TLOAD(t1, r15);
    TSTORE(r0, t6);
    TSTORE(r4, t1);
    TSTORE(r15, t3);
    TSTORE(r0, t2);
    TLOAD(t6, r9);
    TSTORE(r0, t0);
    TSTORE(r15, t5);
    TSTORE(r13, t4);
    TLOAD(t4, r10);
    TLOAD(t5, s9);
    TSTORE(r13, t2);
    TLOAD(t7, r13);
    TLOAD(t6, r5);
    TSTORE(r8, t4);
    TLOAD(t0, r7);
    TLOAD(t4, r4);
    TLOAD(t3, r9);
    TLOAD(t6, r10);
    TLOAD(t4, r2);
    TSTORE(r14, t6);
    TLOAD(t7, s12);
    TLOAD(t2, r8);
    TSTORE(r14, t3);
    TLOAD(t2, s11);
    TSTORE(r3, t6);
    TLOAD(t0, r1);
    TLOAD(t0, r0);
    TLOAD(t2, r3);
    TSTORE(r6, t6);
    TSTORE(r13, t3);
    TSTORE(r7, t3);
    TSTORE(r4, t5);
    TSTORE(r9, t2);
    TLOAD(t3, s13);
    TLOAD(t7, s4);
    TLOAD(t6, s15);
    TLOAD(t3, r7);
    TSTORE(r10, t1);
    TLOAD(t1, r0);
    TSTORE(r10, t0);
    TLOAD(t3, r3);
    TSTORE(r9, t7);
    TSTORE(r2, t5);
    TSTORE(r15, t5);
    TSTORE(r15, t5);
    TLOAD(t3, s9);
    TSTORE(r11, t5);
    TSTORE(r10, t7);
    TSTORE(r5, t5);
    TLOAD(t0, s11);
    TLOAD(t3, s2);
    TSTORE(r0, t7);
    TLOAD(t4, r5);
    TSTORE(r12, t6);
    TLOAD(t6, s6);
    TSTORE(r14, t1);
    TSTORE(r4, t1);
    TLOAD(t3, r0);
    TLOAD(t1, r9);
    BENCHEND;

    TlsuDrain();
    tlsu_finish(1);
    return 0;
}
