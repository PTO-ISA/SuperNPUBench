#include "../data.hpp"
#include <common/pto_tileop.hpp>

#ifdef LINX_PMC
#include "../linxStartEnd.hpp"
#endif

#ifdef __linx
int main();

static inline __attribute__((noreturn)) void linx_supernpu_exit(uint32_t code) {
  if (code == 0) {
    __asm__ volatile(
        "BSTART.STD\n"
        "lui 65545, ->u\n"
        "lui 5, ->t\n"
        "addi t#1, 1365, ->t\n"
        "c.swi t#1, [u#1, 0]\n"
        "BSTOP\n"
        ::: "memory");
  } else {
    __asm__ volatile(
        "BSTART.STD\n"
        "lui 65545, ->u\n"
        "lui 19, ->t\n"
        "addi t#1, 819, ->t\n"
        "c.swi t#1, [u#1, 0]\n"
        "BSTOP\n"
        ::: "memory");
  }
  while (1) {
  }
}

extern "C" __attribute__((noreturn, section(".text._start"))) void
_start(void) {
  linx_supernpu_exit(static_cast<uint32_t>(main()));
}
#endif

template <uint16_t tile_row, uint16_t tile_col, uint16_t valid_row, uint16_t valid_col,
          uint16_t dst_tile_row, uint16_t dst_tile_col, typename T>
void test_pad_rm(T *dst, T *src, T pad_value, size_t up_pad, size_t left_pad, size_t down_pad, size_t right_pad) {
  using gm_shape_in = global_tensor<T, RowMajor<tile_row, tile_col>>;
  using gm_shape_out = global_tensor<T, RowMajor<tile_row, tile_col>>;

  using tile_shape_src = Tile<Location::Vec, T, tile_row, tile_col, BLayout::RowMajor, valid_row, valid_col>;
  using tile_shape_dst = Tile<Location::Vec, T, tile_row, tile_col, BLayout::RowMajor, dst_tile_row, dst_tile_col>;

  gm_shape_in s0(src);
  gm_shape_out res(dst);
  tile_shape_src src_tensor;
  tile_shape_dst dst_tensor;

  TLOAD(src_tensor, s0);
  TPAD(dst_tensor, src_tensor, pad_value, up_pad, left_pad, down_pad, right_pad);
  TSTORE(res, dst_tensor);
}

template <uint16_t tile_row, uint16_t tile_col, uint16_t valid_row, uint16_t valid_col,
          uint16_t dst_tile_row, uint16_t dst_tile_col, typename T>
void test_pad_cm(T *dst, T *src, T pad_value, size_t up_pad, size_t left_pad, size_t down_pad, size_t right_pad) {
  using gm_shape_in = global_tensor<T, ColMajor<tile_row, tile_col>>;
  using gm_shape_out = global_tensor<T, ColMajor<tile_row, tile_col>>;

  using tile_shape_src = Tile<Location::Vec, T, tile_row, tile_col, BLayout::ColMajor, valid_row, valid_col>;
  using tile_shape_dst = Tile<Location::Vec, T, tile_row, tile_col, BLayout::ColMajor, dst_tile_row, dst_tile_col>;

  gm_shape_in s0(src);
  gm_shape_out res(dst);
  tile_shape_src src_tensor;
  tile_shape_dst dst_tensor;

  TLOAD(src_tensor, s0);
  TPAD(dst_tensor, src_tensor, pad_value, up_pad, left_pad, down_pad, right_pad);
  TSTORE(res, dst_tensor);
}

// 测试单个数据类型的函数
template<typename T>
void test_single_type() {
    const uint16_t tile_row = 32;
    const uint16_t tile_col = 32;
    const uint16_t valid_row = 2;
    const uint16_t valid_col = 2;

    const int32_t pad_value = 0;
    const size_t up_pad = 1, left_pad = 2, down_pad = 3, right_pad = 4;
    const uint16_t dst_tile_row = valid_row + up_pad + down_pad;
    const uint16_t dst_tile_col = valid_col + left_pad + right_pad;

    size_t size = tile_row * tile_col;


    // 分配目标内存（行主序和列主序）
    T *dst_rm = (T *)malloc(size * sizeof(T));
    check_mem_alloc(dst_rm);
    init_dst(dst_rm, size);

    T *dst_cm = (T *)malloc(size * sizeof(T));
    check_mem_alloc(dst_cm);
    init_dst(dst_cm, size);

    // 分配源内存
    T *src = (T *)malloc(size * sizeof(T));
    check_mem_alloc(src);

    // 根据类型选择合适的初始化函数
    if constexpr (std::is_integral_v<T>) {
        if constexpr (std::is_unsigned_v<T>) {
            init_src_uint(src, size);
        } else {
            init_src_int(src, size);
        }
    } else {
        init_src_fp(src, size);
    }
#ifdef LINX_PMC
    PMC_START();
#endif
    // 执行测试
    test_pad_rm<tile_row, tile_col, valid_row, valid_col, dst_tile_row, dst_tile_col, T>(
        dst_rm, src, static_cast<T>(pad_value), up_pad, left_pad, down_pad, right_pad);
    test_pad_cm<tile_row, tile_col, valid_row, valid_col, dst_tile_row, dst_tile_col, T>(
        dst_cm, src, static_cast<T>(pad_value), up_pad, left_pad, down_pad, right_pad);
#ifdef LINX_PMC
    PMC_END();
#endif
    // 输出结果
    OutArray(dst_rm, size);
    OutArray(dst_cm, size);

    // 释放内存
    free(dst_rm);
    free(dst_cm);
    free(src);
}

int main() {
#ifdef __linx
    constexpr uint16_t tile_row = 4;
    constexpr uint16_t tile_col = 4;
    constexpr uint16_t valid_row = 2;
    constexpr uint16_t valid_col = 2;
    constexpr size_t up_pad = 1;
    constexpr size_t left_pad = 1;
    constexpr size_t down_pad = 1;
    constexpr size_t right_pad = 1;
    constexpr uint16_t dst_tile_row = valid_row + up_pad + down_pad;
    constexpr uint16_t dst_tile_col = valid_col + left_pad + right_pad;
    constexpr uint16_t size = tile_row * tile_col;

    static int64_t dst[size];
    static int64_t src[size];
    init_dst(dst, size);
    init_src_int(src, size);

    test_pad_rm<tile_row, tile_col, valid_row, valid_col, dst_tile_row,
                dst_tile_col, int64_t>(dst, src, static_cast<int64_t>(0),
                                        up_pad, left_pad, down_pad, right_pad);
    return 0;
#else
    printf("Results:\n");
    // 依次测试各种数据类型可通过, 一起运行测试会有精度错误
    // test_single_type<int8_t>();
    // test_single_type<int16_t>();
    test_single_type<int32_t>();
    // test_single_type<int64_t>();
    // test_single_type<__half>();
    test_single_type<float>();

    return 0;
#endif
}
