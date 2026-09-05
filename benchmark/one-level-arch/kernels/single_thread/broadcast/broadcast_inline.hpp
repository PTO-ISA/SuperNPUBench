// ============================================================================
// Broadcast 算子 — PTO 一层编程模型 (inline 变体: tile 不跨函数传递)
//
// 与 broadcast.hpp 的区别:
//   - 删除 gen_offset_pto 函数模板, 把偏移 tile 计算体 (TCI→TEXPANDS→
//     divmod+stride 累加) 直接内联进 broadcast() 的 full-tile 主循环 与 rmd
//     余数分支 (两处直接重复, 各自用本地 tile_shapeOffset / tile_shapeOffset_rmd
//     类型的 idxTile/coordTile/tmpTile)。
//   - 所有 PTO tile 指令与 tile 对象在同一函数作用域内, 不跨函数传递 tile ——
//     对当前 __vec__ 后端与未来真实 tile-register 内联汇编后端都更稳。
//   - 删除死参数 total_elements / MAX_DIM (原 gen_offset_pto 体内从不引用);
//     base += total_elements 属于 broadcast() 作用域, 保留。
//   - 入口名仍为 broadcast<>; 算法/指令链逐行与 broadcast_pto.hpp 一致。
//
// PTO 一层策略 (与 broadcast_pto.hpp 一致):
//   gen_offset (TCI/TEXPANDS/TREMS/TDIVS/TMULS/TADD) -> MGATHER (按元素索引取数)
//   -> TSTORE; 全部用 Tile 级内联函数, 无 __vec__ 块。
//
// (PTO ISA 指令/编译器状态表 — TCI/TEXPANDS/TREMS/TDIVS/TMULS/TADD/MGATHER/
//  TSTORE 的签名与当前编译器状态说明见 broadcast_pto.hpp 顶部文档注释, 此处不重复。)
// ============================================================================

#include <common/pto_tile.hpp>             // Tile, GlobalTensor 等类型 (当前编译器已有)
#include <common/global_iterator.hpp>      // global_iterator 工具类型 (当前编译器已有)
// #include <pto/pto_instr.hpp>            // [!] PTO ISA C++ Intrinsic — 当前编译器未提供

#include <cstdint>
#include <cstdio>

// ============================================================================
// 维度规则：从后面对齐，前面自动补 1，维度=1 可广播
// ============================================================================

// ----------------------------------------------------------------------------
// broadcast: 接口与 broadcast_pto.hpp 一致 (tile 不跨函数传递)
//
// 偏移 tile 计算体 (原 gen_offset_pto) 内联:
//   1. TCI      生成索引序列  base, base+1, ..., base+N-1
//   2. TEXPANDS 将输出偏移 tile 初始化为 0
//   3. 对每个输出维 d (从 OUT_DIM-1 到 0):
//        TREMS  coord = idx % out_shape[d]          (标量取余)
//        TDIVS  idx   = idx / out_shape[d]          (标量整除)
//        若 d 对应输入维 i = d-(OUT_DIM-IN_DIM) >= 0:
//          非广播维 (in_shape[i]!=1):
//            TMULS  tmp = coord * stride
//            TADD   out += tmp
//          stride *= in_shape[i]
//
// 注意: PTO ISA MGATHER<Coalesce::Elem> 使用元素索引 (非字节偏移),
//       所以这里不再乘 sizeof(dtype)。
// ----------------------------------------------------------------------------
template<typename dtype, size_t MAX_DIM = 8, size_t IN_DIM, size_t OUT_DIM, size_t gIM, size_t gOM, size_t tM>
void broadcast(
    dtype *in_ptr,
    dtype *out_ptr,
    const size_t *in_shape,
    const size_t *out_shape
    ) {
    const size_t Mb = gOM / tM;
    const size_t rmd_M = gOM % tM;

    using gm_shapeIn  = global_tensor<dtype, RowMajor<1, gIM>>;
    using gm_shapeOut = global_tensor<dtype, RowMajor<1, gOM>>;
    using tile_shapeData      = Tile<Location::Vec, dtype,    1, tM, BLayout::RowMajor>;
    using tile_shapeOffset    = Tile<Location::Vec, uint32_t, 1, tM, BLayout::RowMajor>;
    using tile_shapeData_rmd  = Tile<Location::Vec, dtype,    1, tM, BLayout::RowMajor, 1, rmd_M>;
    using tile_shapeOffset_rmd= Tile<Location::Vec, uint32_t, 1, tM, BLayout::RowMajor, 1, rmd_M>;

    gm_shapeIn inGm(in_ptr);
    tile_shapeData outTile;
    tile_shapeOffset offsetTile;
    tile_shapeData_rmd outTile_rmd;
    tile_shapeOffset_rmd offsetTile_rmd;
    size_t base = 0;

    using itOut = global_iterator<gm_shapeOut, tile_shapeData>;
    itOut gOIter(out_ptr);

    size_t total_elements = tM;
    for (int i = 0; i < Mb; ++i) {
        auto gO = gOIter(0, i);

        // ---- offset gen (inlined; tiles stay in broadcast scope) ----
        //   计算 offsetTile (元素索引) 供 MGATHER 使用。
        //   原 gen_offset_pto 函数体, 此处内联, tile 不跨函数传递。
        {
            static_assert(tile_shapeOffset::ValidRow != -1 && tile_shapeOffset::ValidCol != -1,
                          "Only static shape supported");
            using off_t = typename tile_shapeOffset::DType;   // uint32_t
            tile_shapeOffset idxTile;     // 当前正在分解的线性索引
            tile_shapeOffset coordTile;   // 当前输出维坐标
            tile_shapeOffset tmpTile;     // TREMS 需要 (A2A3 要求 tmp >= 1 行, 列数 >= dst)

            // Step 1: TCI 生成索引序列 [base, base+1, ..., base+N-1]
            // [当前编译器] TCI 在 pto_tileop.hpp 有声明，但 jcore 实现为 __vec__ (二层)
            TCI(idxTile, (off_t)base);

            // Step 2: TEXPANDS 初始化 offsetTile = 0
            // [当前编译器] PTO ISA 名 TEXPANDS，当前编译器名 TEXPANDSCALAR；jcore 为 __vec__
            TEXPANDS(offsetTile, (off_t)0);

            size_t stride = 1;

            // Step 3: 从最内维到最外维逐维 divmod + stride 累加
            #pragma clang loop unroll(full)
            for (int d = (int)OUT_DIM - 1; d >= 0; d--) {
                off_t out_d = (off_t)out_shape[d];

                // TREMS: coord = idx % out_d
                // [当前编译器] 完全缺失！pto_tileop.hpp 无 TREMS API。
                //             退路: 用 TDIVS+TMULS+TSUB 三条拼出取余。
                TREMS(coordTile, idxTile, out_d);

                // TDIVS: idx = idx / out_d (推进到下一维)
                // [当前编译器] API 有，jcore 为 __vec__
                TDIVS(idxTile, idxTile, out_d);

                // 该输出维是否对应输入维
                int di = d - (int)(OUT_DIM - IN_DIM);
                if (di >= 0) {
                    if (in_shape[di] != 1) {            // 非广播维才累加
                        // TMULS: tmp = coord * stride
                        // [当前编译器] API 有，jcore 为 __vec__
                        TMULS(tmpTile, coordTile, (off_t)stride);
                        // TADD: out += tmp
                        // [当前编译器] API 有，jcore 为 __vec__
                        TADD(offsetTile, offsetTile, tmpTile);
                    }
                    stride *= in_shape[di];             // stride 更新 (广播维 ==1 不变)
                }
            }
        }
        base += total_elements;

        // MGATHER: 按 offsetTile 中的元素索引从 inGm 取数
        // [当前编译器] template_asm.h 的 MGATHER 已用 asm volatile (一层)，
        //             但不支持 Coalesce::Elem 模板参数；
        //             且旧实现按字节偏移取数，与 PTO ISA 的元素索引语义不同。
        MGATHER(outTile, inGm, offsetTile);

        // TSTORE: 将 outTile 写回 global memory
        // [当前编译器] PTO ISA 名 TSTORE，当前编译器名 TCOPYOUT；jcore 为 __vec__
        TSTORE(gO, outTile);
    }
    if constexpr (rmd_M) {
        auto gO = gOIter(0, Mb);
        total_elements = rmd_M;

        // ---- offset gen (inlined, rmd tiles) ----
        //   与上面 full-tile 计算体逐行一致, 仅 tile 类型换为 *_rmd。
        {
            static_assert(tile_shapeOffset_rmd::ValidRow != -1 && tile_shapeOffset_rmd::ValidCol != -1,
                          "Only static shape supported");
            using off_t = typename tile_shapeOffset_rmd::DType;   // uint32_t
            tile_shapeOffset_rmd idxTile;
            tile_shapeOffset_rmd coordTile;
            tile_shapeOffset_rmd tmpTile;

            TCI(idxTile, (off_t)base);
            TEXPANDS(offsetTile_rmd, (off_t)0);

            size_t stride = 1;
            #pragma clang loop unroll(full)
            for (int d = (int)OUT_DIM - 1; d >= 0; d--) {
                off_t out_d = (off_t)out_shape[d];
                TREMS(coordTile, idxTile, out_d);
                TDIVS(idxTile, idxTile, out_d);
                int di = d - (int)(OUT_DIM - IN_DIM);
                if (di >= 0) {
                    if (in_shape[di] != 1) {
                        TMULS(tmpTile, coordTile, (off_t)stride);
                        TADD(offsetTile_rmd, offsetTile_rmd, tmpTile);
                    }
                    stride *= in_shape[di];
                }
            }
        }
        base += total_elements;
        MGATHER(outTile_rmd, inGm, offsetTile_rmd);
        TSTORE(gO, outTile_rmd);
    }
}
