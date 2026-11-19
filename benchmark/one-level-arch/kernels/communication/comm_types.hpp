#pragma once

#include <cstddef>

namespace supernpu::comm {

enum class Status { ok, invalid_argument, timeout };
constexpr std::size_t kMaxSources = 32;

}  // namespace supernpu::comm
