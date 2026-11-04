// Linx backend mis-materialization of a uniform constant array.
//
// Expected: all four words are 8       -> {8,8,8,8}
// Observed on Linx (both -O0 and -O2): {8,16,8,16}   (odd lanes doubled)
//
// Also reproduced for {1,1,1,1}->{1,2,1,2}, {16,16,16,16}->{16,32,16,32},
// and for a runtime-built uniform array (for-loop writing 9) -> {9,18,9,18}.
// Non-uniform arrays ({1,2,3,4}, {5,6,7,8}) and {0,0,0,0} are correct.
//
// Build (assembly only, for backend inspection):
//   clang++ -O2 -S -mlxbc -fenable-matrix uniform_const_store.cpp -o uniform.s
// Build (runnable, needs the SuperNPUBench harness):
//   see transpose_minrepro.cpp in this dir.

extern "C" volatile unsigned sink[4];

int main() {
    constexpr unsigned s[4] = {8, 8, 8, 8};
    for (int i = 0; i < 4; ++i) {
        sink[i] = s[i];
    }
    return 0;
}
