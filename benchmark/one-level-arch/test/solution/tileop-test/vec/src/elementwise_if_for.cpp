// PTO 0.59 element-wise if smoke test.
//
// This is the scalar-style example from Linx LLVM issue #110/#114.  The
// compiler must keep the loop and lower the condition to a PredicateCell,
// then carry the active mask into both arithmetic arms before TSEL joins the
// results.  In particular, this is not equivalent to computing both arms
// unconditionally and selecting afterwards.
using tile = float tile_size(128);

// Keep the harness storage byte-oriented so the LinxV5 backend does not need
// to synthesize a 128-lane vector initializer before entering the kernel.
alignas(32) static unsigned char lhs_storage[512];
alignas(32) static unsigned char rhs_storage[512];
alignas(32) static unsigned char out_storage[512];

__attribute__((noinline)) static void elementwise_if_for(tile &dst,
                                                         const tile &a,
                                                         const tile &b) {
#pragma linx elementwise
  for (unsigned i = 0; i < 128; ++i) {
    if (a[i] > 0.0f)
      dst[i] = a[i] + b[i];
    else
      dst[i] = a[i] - b[i];
  }
}

__attribute__((optnone)) int main() {
  tile &lhs = *reinterpret_cast<tile *>(lhs_storage);
  tile &rhs = *reinterpret_cast<tile *>(rhs_storage);
  tile &out = *reinterpret_cast<tile *>(out_storage);
  elementwise_if_for(out, lhs, rhs);
  return 0;
}
