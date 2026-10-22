#include "solution/qli/qli_element.hpp"
#include "benchmark.h"
#include "qli_histogram_data.inc"

int main() {
    BENCHSTART;
#include "qli_histogram_calls.inc"
    BENCHEND;
    return 0;
}
