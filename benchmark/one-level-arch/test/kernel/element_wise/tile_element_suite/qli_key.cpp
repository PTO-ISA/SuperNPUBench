#include "solution/qli/qli_element.hpp"
#include "benchmark.h"
#include "qli_key_data.inc"

int main() {
    BENCHSTART;
#include "qli_key_calls.inc"
    BENCHEND;
    return 0;
}
