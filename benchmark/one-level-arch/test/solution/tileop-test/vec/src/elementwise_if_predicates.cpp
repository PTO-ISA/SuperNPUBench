// PTO 0.59 element-wise-if predicate matrix.
//
// Every function keeps the same masked TADD/TSUB arms while changing only the
// predicate.  This makes the generated TCMP mode and predicate inversion easy
// to audit in the disassembly.  main seeds positive, negative, and zero lanes
// so the functional model traverses both arms instead of exercising an
// all-zero static buffer only.
#include <common/pto_tileop.hpp>

#if !defined(PTO_TILEOP_API_HAS_TYPED_ELEMENT_VIEWS) ||                       \
    PTO_TILEOP_API_HAS_TYPED_ELEMENT_VIEWS != 1
#error "elementwise_if_predicates requires the typed element API"
#endif

#include "guard_io.h"

#include <cstddef>

using pto::ElementTile;
using pto::TPARTELEMENT;
using pto::TPARTVIEW;

constexpr std::size_t kElements = 128;
constexpr std::size_t kPartElements = 32;
constexpr unsigned kPredicateCount = 7;
alignas(4096) static float lhs_storage[kElements];
alignas(4096) static float rhs_storage[kElements];
alignas(4096) static float out_storage[kPredicateCount][kElements];

#define EW_IF(NAME, CONDITION)                                               \
  __attribute__((noinline)) static void NAME(                                \
      float *out, const float *lhs, const float *rhs) {                       \
    ElementTile<float, kElements> lhs_tile, rhs_tile;                         \
    TLOAD(lhs_tile, lhs, kElements);                                          \
    TLOAD(rhs_tile, rhs, kElements);                                          \
    auto lhs_parts = TPARTVIEW<kPartElements>(lhs_tile, kElements);           \
    auto rhs_parts = TPARTVIEW<kPartElements>(rhs_tile, kElements);           \
    for (std::size_t part = 0; part < lhs_parts.size(); ++part) {             \
      ElementTile<float, kPartElements> lhs_part, rhs_part, result;           \
      auto lhs_view = lhs_parts.part(part);                                   \
      auto rhs_view = rhs_parts.part(part);                                   \
      TMOV(lhs_part, lhs_view);                                               \
      TMOV(rhs_part, rhs_view);                                               \
      const auto &lhs_elements = TPARTELEMENT(lhs_part);                      \
      const auto &rhs_elements = TPARTELEMENT(rhs_part);                      \
      auto &out_elements = TPARTELEMENT(result);                              \
      _Pragma("pto element for")                                             \
      for (unsigned i = 0; i < kPartElements; ++i) {                          \
        if (CONDITION)                                                        \
          out_elements[i] = lhs_elements[i] + rhs_elements[i];                \
        else                                                                  \
          out_elements[i] = lhs_elements[i] - rhs_elements[i];                \
      }                                                                       \
      TSTORE(out, result, lhs_parts, part);                                   \
    }                                                                         \
  }

EW_IF(if_gt, lhs_elements[i] > 0.0f)
EW_IF(if_lt, lhs_elements[i] < 0.0f)
EW_IF(if_ge, lhs_elements[i] >= 0.0f)
EW_IF(if_le, lhs_elements[i] <= 0.0f)
EW_IF(if_eq, lhs_elements[i] == 0.0f)
EW_IF(if_ne, lhs_elements[i] != 0.0f)
EW_IF(if_not_gt, !(lhs_elements[i] > 0.0f))

__attribute__((optnone)) int main() {
  for (unsigned i = 0; i < kElements; ++i) {
    lhs_storage[i] =
        (i % 3 == 0) ? 1.0f : ((i % 3 == 1) ? -1.0f : 0.0f);
    rhs_storage[i] = 0.5f;
  }
  if_gt(out_storage[0], lhs_storage, rhs_storage);
  if_lt(out_storage[1], lhs_storage, rhs_storage);
  if_ge(out_storage[2], lhs_storage, rhs_storage);
  if_le(out_storage[3], lhs_storage, rhs_storage);
  if_eq(out_storage[4], lhs_storage, rhs_storage);
  if_ne(out_storage[5], lhs_storage, rhs_storage);
  if_not_gt(out_storage[6], lhs_storage, rhs_storage);
  guard_dump_bin(CHK_DIR "/out.bin", out_storage, sizeof(out_storage));
  return 0;
}
