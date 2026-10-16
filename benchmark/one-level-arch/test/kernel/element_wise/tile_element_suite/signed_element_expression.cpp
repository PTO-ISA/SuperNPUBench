#include <common/pto_tileop.hpp>
#include "benchmark.h"
#include "linx_print.h"
#include <cstddef>
#include <cstdint>
#if !defined(PTO_TILEOP_API_HAS_TYPED_ELEMENT_VIEWS)
#error "signed_element_expression requires the official typed element API"
#endif
using pto::ElementTile;
using pto::TPARTVIEW;
using pto::TPARTELEMENT;
constexpr std::size_t kCount=263, kPadded=384, kGuard=16;
constexpr int32_t kGuardValue=-1234567;
extern "C" {
alignas(4096) int32_t signed_element_input[kPadded];
alignas(4096) int32_t signed_element_quotient[kPadded+2*kGuard];
alignas(4096) int32_t signed_element_remainder[kPadded+2*kGuard];
alignas(4096) int32_t signed_element_shift[kPadded+2*kGuard];
alignas(32) uint32_t signed_element_status[8];
}
// 完整 kernel：正式 TileOp 加载/分区，普通 C++ 表达式计算有符号元素，
// 再用 TileOp 写回。程序只描述逻辑元素，不出现物理 layout 或 lane。
__attribute__((noinline)) void signed_element_expression(
    const int32_t *input, std::size_t count, int32_t divisor,
    int32_t *quotient_output, int32_t *remainder_output, int32_t *shift_output) {
  for(std::size_t begin=0;begin<count;begin+=128) {
    const std::size_t valid=count-begin<128?count-begin:128;
    ElementTile<int32_t,128> values;
    TLOAD(values,input+begin,valid);
    auto parts=TPARTVIEW<32>(values,valid);
    for(std::size_t part=0;part<parts.size();++part) {
      auto value_part=parts.part(part);
      ElementTile<int32_t,32> copied, quotient, remainder, shifted;
      TADDS(copied,value_part,int32_t(0));
      auto &elements=TPARTELEMENT(copied);
      auto &quotient_elements=TPARTELEMENT(quotient);
      // 输入范围保证 +19、乘3与减7不溢出。除法必须向零截断。
#pragma pto element for
      for(unsigned element=0;element<32;++element) {
        int32_t biased=elements[element]+19;
        int32_t multiplied=biased*3;
        quotient_elements[element]=(multiplied-7)/divisor;
      }
      TSTORE(quotient_output+begin,quotient,parts,part);
      auto &remainder_elements=TPARTELEMENT(remainder);
      // C++ -7 % 3 == -1；PTO TREM 的结果是 +2，编译器必须保留 C++ 语义。
#pragma pto element for
      for(unsigned element=0;element<32;++element)
        remainder_elements[element]=elements[element]%divisor;
      TSTORE(remainder_output+begin,remainder,parts,part);
      auto &shift_elements=TPARTELEMENT(shifted);
      // 负数右移保留符号；填零后的 padded 元素也参与明确的输出计算。
#pragma pto element for
      for(unsigned element=0;element<32;++element) {
        int32_t right=elements[element]>>2;
        int32_t masked=(right^0x555)&0xff;
        shift_elements[element]=-masked;
      }
      TSTORE(shift_output+begin,shifted,parts,part);
    }
  }
}
int main() {
  for(std::size_t element=0;element<kPadded;++element) {
    int32_t value=-559038737;
    if(element<kCount) value=int32_t((element*7919+12345)%65535)-32767;
    if(element==0) value=-7;
    if(element==1) value=7;
    signed_element_input[element]=value;
  }
  for(std::size_t element=0;element<kPadded+2*kGuard;++element) {
    signed_element_quotient[element]=kGuardValue;
    signed_element_remainder[element]=kGuardValue;
    signed_element_shift[element]=kGuardValue;
  }
  BENCHSTART;
  signed_element_expression(signed_element_input,kCount,3,
      signed_element_quotient+kGuard,signed_element_remainder+kGuard,
      signed_element_shift+kGuard);
  BENCHEND;
  uint32_t failures=0, guards=0;
  for(std::size_t element=0;element<kPadded;++element) {
    const int32_t value=element<kCount?signed_element_input[element]:0;
    failures+=signed_element_quotient[kGuard+element]!=((value+19)*3-7)/3;
    failures+=signed_element_remainder[kGuard+element]!=value%3;
    failures+=signed_element_shift[kGuard+element]!=-(((value>>2)^0x555)&0xff);
  }
  for(std::size_t element=0;element<kGuard;++element) {
    for(std::size_t side=0;side<2;++side) {
      const std::size_t position=side? kGuard+kPadded+element:element;
      guards+=signed_element_quotient[position]!=kGuardValue;
      guards+=signed_element_remainder[position]!=kGuardValue;
      guards+=signed_element_shift[position]!=kGuardValue;
    }
  }
  signed_element_status[0]=kCount;
  signed_element_status[1]=failures+guards;
  signed_element_status[2]=guards;
  signed_element_status[3]=kPadded;
  signed_element_status[4]=uint32_t(signed_element_remainder[kGuard]);
  signed_element_status[5]=uint32_t(signed_element_remainder[kGuard+1]);
  signed_element_status[6]=3;
  signed_element_status[7]=0x53333245; // S32E
  linxi_puts("=== signed_element_expression ===");
  linxi_put_kv("failures",failures+guards);
  return failures+guards?1:0;
}
