#include "deepseekv4.1/layer.hpp"
#include "fixture.hpp"

using namespace deepseek_v41;
extern "C" {
extern test::Input v41_input;
alignas(64) test::Result v41_result;
}

int main() {
    if (get_thread_idx() != 0)
        return 0;
    GlobalMemory mem;
    return test::run(v41_input, v41_result, mem);
}
