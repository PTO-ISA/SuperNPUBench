#!/usr/bin/env python3
"""Exercise the real wrapper with a capturing single-thread kernel boundary."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


class ConcatScatterPartitionTest(unittest.TestCase):
    def test_migrated_and_legacy_allocations(self):
        compiler = shutil.which('c++')
        if compiler is None:
            self.skipTest('host C++ compiler unavailable')
        kernels = Path(__file__).resolve().parents[4] / 'kernels'
        with tempfile.TemporaryDirectory(prefix='pto-concat-partition-') as temp:
            temp = Path(temp)
            (temp / 'common').mkdir()
            (temp / 'basic_op/concat/detail').mkdir(parents=True)
            (temp / 'basic_op/concat/concat_scatter.hpp').write_text(
                (kernels / 'basic_op/concat/concat_scatter.hpp').read_text())
            (temp / 'common/pto_tileop.hpp').write_text('''#pragma once
#include <cstdint>
struct __half {};
inline uint32_t current_pe;
inline uint32_t get_thread_idx() { return current_pe; }
''')
            (temp / 'basic_op/concat/detail/concat_scatter.hpp').write_text('''#pragma once
#include <cstddef>
#include <cstdint>
inline unsigned seen_capacity, seen_offset, seen_count;
template<class T, size_t MaxDim, int IM, int OM, int TM, size_t Rank, size_t Dim>
void concat_scatter(T*, T*, size_t*, size_t*, uint32_t offset,
                    uint32_t count = IM) {
    seen_capacity=IM; seen_offset=offset; seen_count=count;
}
''')
            source = temp / 'test.cpp'
            source.write_text('''#include <cassert>
#include "basic_op/concat/concat_scatter.hpp"
template<class T, int TM, size_t Rank, size_t Dim, int Output, bool Migrated>
void check() {
    T input[4096], output[Output];
    size_t shape[Rank] = {};
    unsigned total=0;
    for (current_pe=0; current_pe<4; ++current_pe) {
        supernpu::multi_thread::concat_scatter<T,8,4096,Output,TM,Rank,Dim>(
            input, output, shape, shape);
        const unsigned workload = Migrated ? Output : 4096;
        assert(seen_capacity == (Migrated ? 4096 : 1024));
        assert(seen_offset == current_pe*(workload/4));
        assert(seen_count == workload/4);
        assert(seen_offset + seen_count <= 4096);
        total += seen_count;
    }
    assert(total == (Migrated ? Output : 4096));
}
int main() {
    check<float,512,2,1,4096,false>();
    check<int32_t,512,3,2,4096,false>();
    check<float,128,2,1,2048,true>();
    check<int32_t,512,2,1,2048,true>();
    check<__half,512,2,1,2048,true>();
}
''')
            binary = temp / 'test'
            subprocess.run([compiler, '-std=c++20', '-I' + str(temp),
                            '-I' + str(kernels), str(source), '-o', str(binary)], check=True)
            subprocess.run([str(binary)], check=True)


if __name__ == '__main__':
    unittest.main()
