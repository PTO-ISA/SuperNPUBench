#include "pto_kernels/memory/tload_store.cpp"

int main() {
  constexpr int elements = pto::kernels::shapes::kMemoryRows *
                           pto::kernels::shapes::kMemoryCols;
  alignas(64) static int source[elements]{};
  alignas(64) static int destination[elements]{};
  tload_store_i32(source, destination);
  return 0;
}
