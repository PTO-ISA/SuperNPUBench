#include "../data.hpp"
#include <common/debug_utils.hpp>
#include <common/pto_tileop.hpp>

#ifdef LINX_PMC
#include "../linxStartEnd.hpp"
#endif

#ifdef ENABLE_TENSOR_INSTR
template <is_tile_data_v tile_shape, is_global_data_v gm_shape>
void TCOPYIN_ASM(tile_shape &dst, gm_shape &src) {

  asm volatile(
    "BSTART.PAR 33, %c1\n"
    "B.ARG ND2NZ.normal, %c1, Null\n"
    "B.IOT [], 0, ->%0<%c2>\n"
    "B.IOR [%3, %4],[]\n"
    "C.B.DIMI %c5, ->lb0\n"
    "C.B.DIMI %c6, ->lb1\n"
    "C.B.DIMI %c7, ->lb2\n"

    ""
    : "=Tr"(dst.data())
    : "i"(type_traits<typename gm_shape::DType>::TypeCode), \
      "i"(tile_type_traits<typename tile_shape::TileDType>::TilesizeCode), \
      "r"(src.data()), "r"(gm_shape::RowStride), "i"(tile_shape::ValidCol), "i"(tile_shape::ValidRow), \
      "i"(tile_shape::Cols)
  );
}
#endif

template <uint16_t gm_row, uint16_t gm_col, uint16_t tile_row,
          uint16_t tile_col, typename T>
void test_Nz(T *dst) {
  using gm_shape = global_tensor<T, RowMajor<gm_row, gm_col>>;
  using tile_shape = TileLeft<T, tile_row, tile_col, 16, 16>;
  gm_shape g(dst);
  tile_shape t;
#ifdef ENABLE_TENSOR_INSTR
  TCOPYIN_ASM(t, g);
#else
  TCOPYIN(t, g);
#endif
  print_tile(t);
}

int main() {
  const uint16_t gm_row = 16;
  const uint16_t gm_col = 16;

  size_t gm_size = gm_row * gm_col;

  float *data = (float *)malloc(gm_size * sizeof(float));
  check_mem_alloc(data);
  init_rows_fp(data, gm_row, gm_col);

  __half *data_hf = (__half *)malloc(gm_size * sizeof(__half));
  check_mem_alloc(data_hf);
  init_rows_fp(data_hf, gm_row, gm_col);

#ifdef LINX_PMC
  PMC_START();
#endif

  test_Nz<gm_row, gm_col, gm_row, gm_col, float>(data);

  test_Nz<gm_row, gm_col, gm_row, gm_col, __half>(data_hf);

#ifdef LINX_PMC
  PMC_END();
#endif
  free(data);
  free(data_hf);
  return 0;
}