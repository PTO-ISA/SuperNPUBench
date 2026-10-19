#include "solution/qli/qli_pto_opt_simple.hpp"
#include "benchmark.h"
#include "qli_topk_data.inc"

int main() {
    BENCHSTART;
#include "qli_topk_calls.inc"
    BENCHEND;
    return 0;
}
