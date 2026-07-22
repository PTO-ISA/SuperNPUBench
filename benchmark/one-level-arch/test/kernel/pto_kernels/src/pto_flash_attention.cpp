#include "pto_kernels/attention/flash_attention.cpp"

int main() {
  constexpr int sequence = pto::kernels::shapes::kAttentionLargeSeq;
  constexpr int query_depth = pto::kernels::shapes::kAttentionSmallQD;
  constexpr int value_depth = pto::kernels::shapes::kAttentionVD;
  alignas(64) static int query[sequence * query_depth]{};
  alignas(64) static int key[query_depth * sequence]{};
  alignas(64) static int value[sequence * value_depth]{};
  alignas(64) static int output[sequence * value_depth]{};
  flash_attention_i32(query, key, value, output);
  return 0;
}
