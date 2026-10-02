// PTO 0.59 element-wise logical-index masked gather/scatter smoke test.
//
// The offset tile contains logical element indices (not byte displacements).
// The LLVM builtin path must preserve the predicate tile as the final B.IOT
// source so disabled lanes do not form memory accesses.
using value_tile = float tile_size(128);
using index_tile = int tile_size(128);
using mask_tile = unsigned char tile_size(128);

alignas(32) static float source_storage[128] =
    {10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f, 70.0f, 80.0f};
alignas(32) static int index_storage[128] = {0, 1, 2, 3, 4, 5, 6, 7};
alignas(32) static unsigned char mask_storage[128] =
    {1, 0, 1, 0, 1, 0, 1, 0};
alignas(32) static float gathered_storage[128];
alignas(32) static float output_storage[128];

__attribute__((noinline)) static void
elementwise_masked_gather_scatter(const value_tile &source,
                                  const index_tile &indices,
                                  const mask_tile &mask) {
#pragma linx elementwise
  value_tile gathered;
  // A 128-lane PTO tile has the logical 16x8 elementwise shape.
  ew_mgather_masked(16, 8, 1, 0, gathered,
                    reinterpret_cast<float *>(source_storage), indices, mask);
  ew_mscatter_masked(16, 8, 1, reinterpret_cast<float *>(output_storage),
                     gathered, indices, mask);
  (void)source;
}

__attribute__((optnone)) int main() {
  value_tile &source = *reinterpret_cast<value_tile *>(source_storage);
  index_tile &indices = *reinterpret_cast<index_tile *>(index_storage);
  mask_tile &mask = *reinterpret_cast<mask_tile *>(mask_storage);
  (void)output_storage;
  (void)gathered_storage;
  elementwise_masked_gather_scatter(source, indices, mask);
  return 0;
}
