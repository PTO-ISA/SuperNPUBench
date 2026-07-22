#include "pto_kernels/elementwise/add_custom.cpp"

int main() {
  constexpr int elements = pto::kernels::shapes::kMemoryRows *
                           pto::kernels::shapes::kMemoryCols;
  alignas(64) static float x[elements]{};
  alignas(64) static float y[elements]{};
  alignas(64) static float z[elements]{};
  add_custom_f32(x, y, z);
  return 0;
}
