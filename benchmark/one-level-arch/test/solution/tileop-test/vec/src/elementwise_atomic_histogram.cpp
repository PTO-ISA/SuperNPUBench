// PTO 0.59 element-wise atomic histogram smoke test.
//
// Both indexed operations use logical element indices.  Every lane adds one
// to a deliberately repeated bin index, so a non-atomic lowering loses
// updates.  The result is read back with masked MGATHER to exercise the full
// atomic-write + indexed-load flow.
using value_tile = int tile_size(128);
using index_tile = int tile_size(128);
using mask_tile = unsigned char tile_size(128);

alignas(32) static int bins[8];
alignas(32) static int values[128];
alignas(32) static int indices[128];
alignas(32) static unsigned char active[128];
alignas(32) static unsigned int observed[128];

__attribute__((noinline)) static void elementwise_atomic_histogram(
    const value_tile &value, const index_tile &index, const mask_tile &mask) {
#pragma linx elementwise
  value_tile gathered;
  ew_mscatter_add(16, 8, 17, reinterpret_cast<unsigned int *>(bins), value,
                  index);
  ew_mgather_masked(16, 8, 17, 0, gathered,
                    reinterpret_cast<unsigned int *>(bins), index, mask);
  (void)gathered;
  (void)mask;
}

__attribute__((optnone)) int main() {
  for (unsigned int lane = 0; lane < 128; ++lane) {
    bins[lane < 8 ? lane : 0] = 0;
    values[lane] = 1;
    indices[lane] = lane & 7u;
    active[lane] = 1;
    observed[lane] = 0;
  }
  value_tile &value = *reinterpret_cast<value_tile *>(values);
  index_tile &index = *reinterpret_cast<index_tile *>(indices);
  mask_tile &mask = *reinterpret_cast<mask_tile *>(active);
  elementwise_atomic_histogram(value, index, mask);
  return 0;
}
