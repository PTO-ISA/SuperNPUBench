#include <cstdint>

#include "benchmark.h"
#include "gm_atom_dup_driver.h"

namespace {
gm_atom_dup::Buffers buffers{};
MultiThreadResCheckSync res_check_sync{};
}  // namespace

int main() {
    const std::uint32_t tid = get_thread_idx();
    gm_atom_dup::prepare(buffers, res_check_sync, tid);

    BENCHSTART;
    gm_atom_dup::run(buffers, tid);
    BENCHEND;

    return gm_atom_dup::finish(buffers, res_check_sync, tid);
}
