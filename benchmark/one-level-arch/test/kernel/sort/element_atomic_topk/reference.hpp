#ifndef ELEMENT_ATOMIC_TOPK_REFERENCE_HPP
#define ELEMENT_ATOMIC_TOPK_REFERENCE_HPP

#include <cstddef>
#include <cstdint>

namespace element_atomic_topk {

constexpr std::size_t kRadix = 256;

struct Cutoff {
  uint16_t key;
  uint32_t strictly_greater;
  uint32_t equal_needed;
};

inline void clear_histogram(uint32_t (&hist)[kRadix]) {
  for (std::size_t i = 0; i < kRadix; ++i) {
    hist[i] = 0;
  }
}

inline void high8_histogram(const uint32_t *keys, std::size_t count,
                            uint32_t (&hist)[kRadix]) {
  clear_histogram(hist);
  for (std::size_t i = 0; i < count; ++i) {
    ++hist[(keys[i] >> 8) & 0xffu];
  }
}

inline uint8_t find_descending_bucket(const uint32_t (&hist)[kRadix],
                                      std::size_t k,
                                      std::size_t *greater_count) {
  std::size_t greater = 0;
  if (k == 0) {
    *greater_count = 0;
    return 255;
  }
  for (int bucket = 255; bucket >= 0; --bucket) {
    const std::size_t next = greater + hist[bucket];
    if (next >= k) {
      *greater_count = greater;
      return static_cast<uint8_t>(bucket);
    }
    greater = next;
  }
  *greater_count = greater;
  return 0;
}

inline void selected_low8_histogram(const uint32_t *keys, std::size_t count,
                                    uint8_t selected_high,
                                    uint32_t (&hist)[kRadix]) {
  clear_histogram(hist);
  for (std::size_t i = 0; i < count; ++i) {
    const uint32_t key = keys[i];
    if (((key >> 8) & 0xffu) == selected_high) {
      ++hist[key & 0xffu];
    }
  }
}

inline Cutoff find_cutoff(const uint32_t *keys, std::size_t count,
                          std::size_t requested_k,
                          uint32_t (&high_hist)[kRadix],
                          uint32_t (&low_hist)[kRadix]) {
  const std::size_t k = requested_k < count ? requested_k : count;
  high8_histogram(keys, count, high_hist);
  if (k == 0) {
    clear_histogram(low_hist);
    return Cutoff{UINT16_MAX, 0, 0};
  }

  std::size_t greater_high = 0;
  const uint8_t high = find_descending_bucket(high_hist, k, &greater_high);
  selected_low8_histogram(keys, count, high, low_hist);

  std::size_t greater_low = 0;
  const uint8_t low =
      find_descending_bucket(low_hist, k - greater_high, &greater_low);
  const std::size_t strictly_greater = greater_high + greater_low;
  return Cutoff{static_cast<uint16_t>((static_cast<uint16_t>(high) << 8) |
                                      static_cast<uint16_t>(low)),
                static_cast<uint32_t>(strictly_greater),
                static_cast<uint32_t>(k - strictly_greater)};
}

inline std::size_t collect_topk(const uint32_t *keys, std::size_t count,
                                std::size_t requested_k, const Cutoff &cutoff,
                                uint32_t *output) {
  const std::size_t k = requested_k < count ? requested_k : count;
  std::size_t output_count = 0;
  for (std::size_t i = 0; i < count; ++i) {
    const uint16_t key = static_cast<uint16_t>(keys[i]);
    if (key > cutoff.key) {
      output[output_count++] = key;
    }
  }
  std::size_t equal_remaining = cutoff.equal_needed;
  for (std::size_t i = 0; i < count && equal_remaining != 0; ++i) {
    const uint16_t key = static_cast<uint16_t>(keys[i]);
    if (key == cutoff.key) {
      output[output_count++] = key;
      --equal_remaining;
    }
  }
  return output_count == k && equal_remaining == 0 ? output_count : 0;
}

inline std::size_t reference_topk(const uint32_t *keys, std::size_t count,
                                  std::size_t requested_k, uint32_t *output,
                                  uint32_t (&high_hist)[kRadix],
                                  uint32_t (&low_hist)[kRadix],
                                  Cutoff *cutoff_out = nullptr) {
  const Cutoff cutoff =
      find_cutoff(keys, count, requested_k, high_hist, low_hist);
  if (cutoff_out != nullptr) {
    *cutoff_out = cutoff;
  }
  return collect_topk(keys, count, requested_k, cutoff, output);
}

} // namespace element_atomic_topk

#endif
