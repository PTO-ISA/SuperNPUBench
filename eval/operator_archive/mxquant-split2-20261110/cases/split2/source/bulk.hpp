#pragma once
#include "baseline.hpp"

namespace mxquant_bulk {
using namespace pto;
using Input = VecTileM32<__bf16,32,256>;
using Output = VecTileM32<__fp8_e4m3,32,256>;
using Parts = TileArray<mxquant::QuantBlockTile,1,8>;

// The original algorithm, with the two x consumers borrowing the loaded parent.
// No arithmetic is inserted to materialize a subview.
template<class View>
__attribute__((always_inline)) inline void group(
    mxquant::ScaleTile &scale, Parts &parts, View &x, int g) {
  mxquant::BlockTile ax, ax_scaled, scaled;
  mxquant::RowMaxWideTile wide;
  TABS(ax,x);
  TMULS(ax_scaled,ax,mxquant::kInv256);
  TROWMAX(wide,ax_scaled);
  auto prefix=TREDUCEPREFIXVIEW<mxquant::RowMaxTile>(wide);
  TCVT<LINX_RDN>(scale,prefix);
  mxquant::RowMaxTile scale_bf16, reciprocal;
  TCVT(scale_bf16,scale);
  TRECIP(reciprocal,scale_bf16);
  TROWEXPANDMUL(scaled,x,reciprocal);
  TCVT(parts[0][g],scaled);
  asm volatile("" ::: "memory");
}

__attribute__((always_inline)) inline void load(Input &x,const __bf16 *p) {
  global_tensor<__bf16,RowMajor<32,256>> gm(p);
  TLOAD(x,gm);
  asm volatile("" ::: "memory");
}

using Word=VecTileM32<uint32_t,32,1>;
using ScaleWords=VecTileM32<uint32_t,32,2>;
using ScaleParts=TileArray<Word,1,2>;
__attribute__((always_inline)) inline void pack32(
    Word &dst,Word &left,Word &right,uint64_t control) {
  // RMS runtime rejects the public TPACK stream (Block.cpp:3220); trying
  // an early-clobber output also hits an existing compiler backend limit.
  // Both fields are already zero-extended U32, so shift+OR is equivalent.
  // This workaround costs 12 ops instead of 6 TPACK ops per row band.
  Word shifted;
  TSHLS(shifted,right,static_cast<uint32_t>((control&255)*8));
  TOR(dst,left,shifted);
}
__attribute__((always_inline)) inline void pack_scales(
    __fp8_e8m0 *dst,mxquant::ScaleTile &c0,mxquant::ScaleTile &c1,
    mxquant::ScaleTile &c2,mxquant::ScaleTile &c3,mxquant::ScaleTile &c4,
    mxquant::ScaleTile &c5,mxquant::ScaleTile &c6,mxquant::ScaleTile &c7) {
  // TPACK accepts unsigned integer backing, not E8M0. TCVT's same-width
  // source-operation view is legal (PTO-TILE-CARRIER-REINTERPRETATION-001):
  // interpret the unchanged E8M0 byte as U8, numerically widen it to U32.
  // This creates genuinely U32-backed words; it never retags an E8M0 tile.
  auto v0=reinterpret_tile<uint8_t>(c0); Word w0; TCVT(w0,v0);
  auto v1=reinterpret_tile<uint8_t>(c1); Word w1; TCVT(w1,v1);
  auto v2=reinterpret_tile<uint8_t>(c2); Word w2; TCVT(w2,v2);
  auto v3=reinterpret_tile<uint8_t>(c3); Word w3; TCVT(w3,v3);
  auto v4=reinterpret_tile<uint8_t>(c4); Word w4; TCVT(w4,v4);
  auto v5=reinterpret_tile<uint8_t>(c5); Word w5; TCVT(w5,v5);
  auto v6=reinterpret_tile<uint8_t>(c6); Word w6; TCVT(w6,v6);
  auto v7=reinterpret_tile<uint8_t>(c7); Word w7; TCVT(w7,v7);
  Word p01,p23,p45,p67,lo,hi;
  pack32(p01,w0,w1,0x0101); pack32(p23,w2,w3,0x0101);
  pack32(p45,w4,w5,0x0101); pack32(p67,w6,w7,0x0101);
  pack32(lo,p01,p23,0x0202); pack32(hi,p45,p67,0x0202);
  // Public TileArray assembly has a TCVT producer, but no TPACK producer.
  // Two identity TCVTs materialize the packed halves into the assembled
  // parent: explicit cost = 8 widen + 12 shift/OR + 2 identity ops/band.
  ScaleParts parts; TCVT(parts[0][0],lo); TCVT(parts[0][1],hi);
  auto all=TASSEMBLY<ScaleWords>(static_cast<ScaleParts&&>(parts));
  // GM contains raw scale bytes. The destination type matches the actual
  // U32 register backing; no E8M0 tile reinterpretation is used for TSTORE.
  global_tensor<uint32_t,RowMajor<32,2>> gm(reinterpret_cast<uint32_t*>(dst));
  TSTORE(gm,all);
  asm volatile("" ::: "memory");
}

template<bool Packed>
__attribute__((always_inline)) inline void block(
    Input &x,__fp8_e4m3 *out,__fp8_e8m0 *scales) {
  auto views=TPARTVIEW<mxquant::BlockTile,1,8>(x);
  Parts parts;
  mxquant::ScaleTile c0,c1,c2,c3,c4,c5,c6,c7;
  auto v0=views[0][0]; group(c0,parts,v0,0);
  auto v1=views[0][1]; group(c1,parts,v1,1);
  auto v2=views[0][2]; group(c2,parts,v2,2);
  auto v3=views[0][3]; group(c3,parts,v3,3);
  auto v4=views[0][4]; group(c4,parts,v4,4);
  auto v5=views[0][5]; group(c5,parts,v5,5);
  auto v6=views[0][6]; group(c6,parts,v6,6);
  auto v7=views[0][7]; group(c7,parts,v7,7);
  auto q=TASSEMBLY<Output>(static_cast<Parts&&>(parts));
  global_tensor<__fp8_e4m3,RowMajor<32,256>> gmout(out);
  TSTORE(gmout,q);
  asm volatile("" ::: "memory");
  if constexpr(Packed) {
    pack_scales(scales,c0,c1,c2,c3,c4,c5,c6,c7);
  } else {
  using SG=global_tensor<__fp8_e8m0,RowMajor<32,8>>;
  SG s0(scales+0); TSTORE(s0,c0);
  SG s1(scales+1); TSTORE(s1,c1);
  SG s2(scales+2); TSTORE(s2,c2);
  SG s3(scales+3); TSTORE(s3,c3);
  SG s4(scales+4); TSTORE(s4,c4);
  SG s5(scales+5); TSTORE(s5,c5);
  SG s6(scales+6); TSTORE(s6,c6);
  SG s7(scales+7); TSTORE(s7,c7);
  asm volatile("" ::: "memory");
  }
}

template<bool PingPong,bool Packed=false>
inline void run(__fp8_e4m3 *out,__fp8_e8m0 *scales,const __bf16 *in) {
  auto tid=get_thread_idx();
  if(tid>=4)return;
  in+=tid*128*256; out+=tid*128*256; scales+=tid*128*8;
  if constexpr(!PingPong) {
    for(int rb=0;rb<4;++rb) {
      Input x; load(x,in+rb*32*256);
      block<Packed>(x,out+rb*32*256,scales+rb*32*8);
    }
  } else {
    // Two independently named parent carriers; prefetch the next row block
    // before consuming the current one, then recycle only after its consumers.
    Input ping,pong;
    load(ping,in);
    load(pong,in+32*256);
    block<Packed>(ping,out,scales);
    load(ping,in+2*32*256);
    block<Packed>(pong,out+32*256,scales+32*8);
    load(pong,in+3*32*256);
    block<Packed>(ping,out+2*32*256,scales+2*32*8);
    block<Packed>(pong,out+3*32*256,scales+3*32*8);
  }
}
}
