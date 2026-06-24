#include "../data.hpp"
#include <common/pto_tileop.hpp>

#ifdef LINX_PMC
#include "../linxStartEnd.hpp"
#endif

template <uint16_t srci_row, uint16_t srci_col, uint16_t dst_row,
          uint16_t dst_col>
void test(float *dst, uint16_t *srci, float s) {
  using gm_shape_srci = global_tensor<uint16_t, RowMajor<srci_row, srci_col>>;
  using gm_shape_dst = global_tensor<float, RowMajor<dst_row, dst_col>>;

  using tile_shape_srci = Tile<Location::Vec, uint16_t, srci_row, srci_col, BLayout::RowMajor>;
  using tile_shape_dst = Tile<Location::Vec, float, dst_row, dst_col, BLayout::RowMajor>;

  gm_shape_srci s0(srci);
  gm_shape_dst s1(dst);
  gm_shape_dst res(dst);

  tile_shape_srci d0;
  tile_shape_dst d1;

  TLOAD(d0, s0);
  TLOAD(d1, s1);
  TSCATTERELEMENTROW(d1, d0, s);
  TSTORE(res, d1);
}

int main() {
  const uint16_t srci_row = 16;
  const uint16_t srci_col = 64;
  const uint16_t dst_row = 128;
  const uint16_t dst_col = 64;

  size_t size_srci = srci_row * srci_col;
  size_t size_dst = dst_row * dst_col;

  float *dst = (float *)malloc(size_dst * sizeof(float));
  check_mem_alloc(dst);
  init_dst(dst, size_dst);

  uint16_t *srci = (uint16_t *)malloc(size_srci * sizeof(uint16_t));
  check_mem_alloc(srci);
  init_index(srci, srci_row, srci_col);

  float s = s_fp32;

#ifdef LINX_PMC
  PMC_START();
#endif

  test<srci_row, srci_col, dst_row, dst_col>(dst, srci, s);

#ifdef LINX_PMC
  PMC_END();
#endif

  printf("Result:\n");
  OutArray(dst, size_dst);

  free(dst);
  free(srci);

  return 0;
}