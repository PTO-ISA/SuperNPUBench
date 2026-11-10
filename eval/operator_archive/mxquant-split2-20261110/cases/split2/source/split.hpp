#pragma once
#include "bulk.hpp"

namespace mxquant_split {
using namespace mxquant_bulk;
template<int Chunks> using Chunk=VecTileM32<__bf16,32,256/Chunks>;
template<int Chunks> struct Inputs { Chunk<Chunks> a,b,c,d; };

template<int Chunks>
__attribute__((always_inline)) inline void load_chunk(Chunk<Chunks> &x,const __bf16 *p) {
  // Logical rows still have 256 BF16 elements. Each request covers a subset
  // of columns; unlike the original full-width load it is row-strided.
  global_tensor<__bf16,RowMajor<32,256>> gm(p);
  TLOAD(x,gm);
  asm volatile("" ::: "memory");
}
template<int Chunks>
__attribute__((always_inline)) inline void load(Inputs<Chunks> &x,const __bf16 *p) {
  load_chunk<Chunks>(x.a,p);load_chunk<Chunks>(x.b,p+256/Chunks);
  if constexpr(Chunks==4) {
    load_chunk<Chunks>(x.c,p+128);load_chunk<Chunks>(x.d,p+192);
  }
}
template<int Chunks>
__attribute__((always_inline)) inline void block(Inputs<Chunks> &x,__fp8_e4m3 *out,__fp8_e8m0 *scales) {
  auto va=TPARTVIEW<mxquant::BlockTile,1,8/Chunks>(x.a);
  auto vb=TPARTVIEW<mxquant::BlockTile,1,8/Chunks>(x.b);
  Parts parts;mxquant::ScaleTile c0,c1,c2,c3,c4,c5,c6,c7;
  auto v0=va[0][0];group(c0,parts,v0,0);
  auto v1=va[0][1];group(c1,parts,v1,1);
  if constexpr(Chunks==2) {
    auto v2=va[0][2];group(c2,parts,v2,2);
    auto v3=va[0][3];group(c3,parts,v3,3);
    auto v4=vb[0][0];group(c4,parts,v4,4);
    auto v5=vb[0][1];group(c5,parts,v5,5);
    auto v6=vb[0][2];group(c6,parts,v6,6);
    auto v7=vb[0][3];group(c7,parts,v7,7);
  } else {
    auto vc=TPARTVIEW<mxquant::BlockTile,1,2>(x.c);
    auto vd=TPARTVIEW<mxquant::BlockTile,1,2>(x.d);
    auto v2=vb[0][0];group(c2,parts,v2,2);
    auto v3=vb[0][1];group(c3,parts,v3,3);
    auto v4=vc[0][0];group(c4,parts,v4,4);
    auto v5=vc[0][1];group(c5,parts,v5,5);
    auto v6=vd[0][0];group(c6,parts,v6,6);
    auto v7=vd[0][1];group(c7,parts,v7,7);
  }
  auto q=TASSEMBLY<Output>(static_cast<Parts&&>(parts));
  global_tensor<__fp8_e4m3,RowMajor<32,256>> gmout(out);TSTORE(gmout,q);
  asm volatile("" ::: "memory");
  pack_scales(scales,c0,c1,c2,c3,c4,c5,c6,c7);
}
template<int Chunks>
inline void run(__fp8_e4m3 *out,__fp8_e8m0 *scales,const __bf16 *in) {
  auto tid=get_thread_idx();if(tid>=4)return;
  in+=tid*128*256;out+=tid*128*256;scales+=tid*128*8;
  Inputs<Chunks> ping,pong;
  load(ping,in);load(pong,in+32*256);
  block(ping,out,scales);
  load(ping,in+2*32*256);block(pong,out+32*256,scales+32*8);
  load(pong,in+3*32*256);block(ping,out+2*32*256,scales+2*32*8);
  block(pong,out+3*32*256,scales+3*32*8);
}
}
