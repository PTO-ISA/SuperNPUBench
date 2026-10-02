// PTO 0.59 element-wise-if predicate matrix.
//
// Every function keeps the same masked TADD/TSUB arms while changing only the
// predicate.  This makes the generated TCMP mode and predicate inversion easy
// to audit in the disassembly.  main seeds positive, negative, and zero lanes
// so the functional model traverses both arms instead of exercising an
// all-zero static buffer only.
using tile = float tile_size(128);

alignas(32) static unsigned char lhs_storage[512];
alignas(32) static unsigned char rhs_storage[512];
alignas(32) static unsigned char out_storage[512];

#define EW_IF(NAME, CONDITION)                                               \
  __attribute__((noinline)) static void NAME(tile &out, const tile &lhs,    \
                                              const tile &rhs) {             \
  _Pragma("linx elementwise")                                               \
    for (unsigned i = 0; i < 128; ++i) {                                    \
      if (CONDITION)                                                         \
        out[i] = lhs[i] + rhs[i];                                            \
      else                                                                   \
        out[i] = lhs[i] - rhs[i];                                            \
    }                                                                        \
  }

EW_IF(if_gt, lhs[i] > 0.0f)
EW_IF(if_lt, lhs[i] < 0.0f)
EW_IF(if_ge, lhs[i] >= 0.0f)
EW_IF(if_le, lhs[i] <= 0.0f)
EW_IF(if_eq, lhs[i] == 0.0f)
EW_IF(if_ne, lhs[i] != 0.0f)
EW_IF(if_not_gt, !(lhs[i] > 0.0f))

__attribute__((optnone)) int main() {
  tile &lhs = *reinterpret_cast<tile *>(lhs_storage);
  tile &rhs = *reinterpret_cast<tile *>(rhs_storage);
  tile &out = *reinterpret_cast<tile *>(out_storage);
  float *lhs_values = reinterpret_cast<float *>(lhs_storage);
  float *rhs_values = reinterpret_cast<float *>(rhs_storage);
  for (unsigned i = 0; i < 128; ++i) {
    lhs_values[i] = (i % 3 == 0) ? 1.0f : ((i % 3 == 1) ? -1.0f : 0.0f);
    rhs_values[i] = 0.5f;
  }
  if_gt(out, lhs, rhs);
  if_lt(out, lhs, rhs);
  if_ge(out, lhs, rhs);
  if_le(out, lhs, rhs);
  if_eq(out, lhs, rhs);
  if_ne(out, lhs, rhs);
  if_not_gt(out, lhs, rhs);
  return 0;
}
