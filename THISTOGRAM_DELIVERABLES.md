# THISTOGRAM — 5 Required Deliverables

## 1. Function Signature
File: include/jcore/template_asm.hpp:224
```cpp
template <is_tile_data_v tile_shape_out, is_tile_data_v tile_shape_in>
void THISTOGRAM(tile_shape_out &dst, tile_shape_in &src, tile_shape_in &Idx, int ByteId)
```

## 2. Idx Shape Constraints
File: emulator/engine/AccumulateBlockInfo.cpp:201-207
- UINT32: Idx.validRow >= 3 - ByteId, Idx.validCol >= 1
- UINT16+ByteId=0: Idx.validRow >= src.validRow, Idx.validCol >= 1
- Conclusion: Idx does NOT need to match src shape

## 3. Assembly Template
File: include/jcore/template_asm.hpp:226-233
```
BSTART.TEPL 0b1101000, %c1
B.DATR %c2, ByteN, Null
B.DIM %3, 0, ->LB0
B.DIM %4, 0, ->LB1
B.DIM zero, %c5, ->LB2
B.IOT %6, %7, mask=15, last, ->%0<%Z8>
""
```

## 4. Related Histogram APIs
File: include/jcore/template_asm.hpp
- TSORT (line 6711)
- TMRGSORT (line 6793)
- TPARTADD (line 6908)
- TPARTMUL (line 6929)
- TPARTMAX (line 6950)
- TPARTMIN (line 6971)

## 5. Non-qli Reference Code
Files: microbenchmark/vector/src/thistogram_i32_16x16.cpp:10, thistogram_i16_16x16.cpp:10
```cpp
bench_hist_u32<int32_t,16,16>(ch,a,b,0,[](auto& dst,auto& s,auto& idx,auto b){ THISTOGRAM(dst,s,idx,b); });
```
