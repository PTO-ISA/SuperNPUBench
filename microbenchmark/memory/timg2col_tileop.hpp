#ifndef MICROBENCHMARK_TIMG2COL_TILEOP_HPP
#define MICROBENCHMARK_TIMG2COL_TILEOP_HPP

#include <common/pto_tileop.hpp>
#include <cstdint>
#include <type_traits>

namespace microbench {

// PTO-BSTART-TIMG2COL-PARAMS-001 parameter-word packing.
constexpr uint64_t PackTIMG2COLParam0(uint16_t input_h, uint16_t input_w,
                                      uint16_t cin, uint8_t kernel_h,
                                      uint8_t kernel_w) {
  return static_cast<uint64_t>(input_h) |
         (static_cast<uint64_t>(input_w) << 16) |
         (static_cast<uint64_t>(cin) << 32) |
         (static_cast<uint64_t>(kernel_h) << 48) |
         (static_cast<uint64_t>(kernel_w) << 56);
}

constexpr uint64_t PackTIMG2COLParam1(uint8_t pad_top, uint8_t pad_left,
                                      uint8_t pad_bottom, uint8_t pad_right,
                                      uint8_t dilation_h, uint8_t dilation_w,
                                      uint8_t stride_h, uint8_t stride_w) {
  return static_cast<uint64_t>(pad_top) |
         (static_cast<uint64_t>(pad_left) << 8) |
         (static_cast<uint64_t>(pad_bottom) << 16) |
         (static_cast<uint64_t>(pad_right) << 24) |
         ((static_cast<uint64_t>(dilation_h) & 0x1f) << 32) |
         ((static_cast<uint64_t>(dilation_w) & 0x1f) << 37) |
         ((static_cast<uint64_t>(stride_h) & 0x3f) << 42) |
         ((static_cast<uint64_t>(stride_w) & 0x3f) << 48);
}

constexpr uint64_t PackTIMG2COLParam2(uint32_t row_start,
                                      uint32_t col_start) {
  return static_cast<uint64_t>(row_start) |
         (static_cast<uint64_t>(col_start) << 32);
}

constexpr bool IsTIMG2COLDataType(int type_code) {
  return type_code == __type_fp32 || type_code == __type_tf32 ||
         type_code == __type_hf32 || type_code == __type_fp16 ||
         type_code == __type_bf16 || type_code == __type_hif8 ||
         type_code == __type_fp8_e4m3 || type_code == __type_fp8_e5m2 ||
         type_code == __type_fp8_e8m0 || type_code == __type_int32 ||
         type_code == __type_int16 || type_code == __type_int8 ||
         type_code == __type_uint32 || type_code == __type_uint16 ||
         type_code == __type_uint8;
}

// Direct Local-CUBE form of PTO ISA 0.58.6 BSTART.TIMG2COL.  This is a
// microbenchmark-local wrapper intentionally expressed as inline assembly;
// it does not call the installed TileOP TIMG2COL implementation.
template <int GroupRows, typename TileOut, typename GlobalIn>
inline void TIMG2COL_ASM(TileOut &dst, GlobalIn &src, uint64_t param0,
                         uint64_t param1, uint64_t param2) {
  static_assert(GroupRows > 0 && GroupRows <= 128,
                "TIMG2COL group ValidRow must be in [1, 128]");
  static_assert(TileOut::Loc == pto::Location::Left,
                "TIMG2COL direct output must use the Left location");
  static_assert(TileOut::BFractal == pto::BLayout::CubeM16 ||
                    TileOut::BFractal == pto::BLayout::CubeM32,
                "TIMG2COL direct output must use CUBE_M16 or CUBE_M32");
  static_assert(TileOut::ValidRow > 0 && TileOut::ValidCol > 0,
                "TIMG2COL destination dimensions must be static and nonzero");
  static_assert(std::is_same_v<typename TileOut::DType,
                               typename GlobalIn::DType>,
                "TIMG2COL source and destination dtypes must match");
  static_assert(IsTIMG2COLDataType(
                    ::type_traits<typename GlobalIn::DType>::TypeCode),
                "TIMG2COL dtype is not accepted by PTO ISA 0.58.6");

  // Keep zero-valued parameter words in allocated GPRs.  TIMG2COL requires
  // exactly two contiguous source-only B.IOR records rather than folded or
  // omitted operands.
  asm("" : "+r"(param0));
  asm("" : "+r"(param1));
  asm("" : "+r"(param2));

  // The main compiler at llvm 553b08045 does not yet parse the new standalone
  // mnemonic. Emit its normative L32 word directly: low 27 bits 0x01c11181,
  // with DataType in bits [31:27]. The remaining bundle modifiers still use
  // normal assembly operands so register allocation stays compiler-managed.
  constexpr uint32_t bstart_word =
      0x01c11181u |
      (static_cast<uint32_t>(
           ::type_traits<typename GlobalIn::DType>::TypeCode)
       << 27);
  // B.DATR has DTYPE_NONE=31, PadValue=Zero, all controls zero, and the
  // source-to-destination Layout in bits [11:7]. The current assembler also
  // rejects TIMG2COL's legal ND2M16/ND2M32 selectors, so emit this normative
  // modifier word directly as well.
  constexpr uint32_t layout =
      TileOut::BFractal == pto::BLayout::CubeM16 ? 22u : 21u;
  constexpr uint32_t datr_word = 0x01f01023u | (layout << 7);

  asm volatile(
      ".4byte %c[BStartWord]\n"
      ".4byte %c[DAttrWord]\n"
      "B.DIM zero, %c[ValidCol], ->lb0\n"
      "B.DIM zero, %c[GroupRows], ->lb1\n"
      "B.DIM zero, %c[TotalCol], ->lb2\n"
      "B.IOR [%[GMBase]], []\n"
      "B.IOR [%[Param0], %[Param1], %[Param2]], []\n"
      "B.IOT mask=1111, last, ->%[Dst]<%Z[TileSize]>\n"
      : [Dst] "=Tr"(dst.data())
      : [BStartWord] "i"(bstart_word), [DAttrWord] "i"(datr_word),
        [GMBase] "r"(src.data()), [ValidCol] "i"(TileOut::ValidCol),
        [GroupRows] "i"(GroupRows), [TotalCol] "i"(TileOut::Cols),
        [Param0] "r"(param0), [Param1] "r"(param1),
        [Param2] "r"(param2),
        [TileSize] "i"(
            ::tile_type_traits<typename TileOut::TileDType>::TilesizeCode)
      : "memory");
}

} // namespace microbench

#endif
