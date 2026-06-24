#include "../common/data.hpp"
#include <common/pto_tileop.hpp>
#include "acl/acl.h"

template <const AS as_, typename T,
        int Rows_, int Cols_, const FractalLayoutKind fractal_, const int active_cols_, const int active_rows_
        const int fractalSize_ = 512>
__global__[aicore] void test_tadd(__gm__ T *out, __gm__ T *src0, __gm__ T *src1) {
    using tile_shape = Tile<as_, T, RowMajor<Rows_, Cols_>, fractal_, active_cols_, active_rows_>;
    tile_shape src0_tile, src1_tile, dst_tile;
    src0_tile.assignAddr(0x0);
    src1_tile.assignAddr(0x4000);
    dst_tile.assignAddr(0x8000);
    PTO::TLOAD<tile_shape>(src0_tile, src0, 64);
    PTO::TLOAD<tile_shape>(src1_tile, src1, 64);
    PTO::TADD<tile_shape>(dst_tile, src0_tile, src1_tile);
    PTO::TSTORE<tile_shape>(out, dst_tile);
}

int main() {
    aclInit(nullptr);
    aclrtSetDevice(0);

    aclrtStream stream;
    aclrtCreateStream(&stream);

    std::vector<int> shape = {64, 64};
    int capacity = shape[0] * shape[1];
    int byteSize = shape[0] * shape[1] * sizeof(uint8_t);

    uint8_t *dstHost, *src0Host, *src1Host;
    uint8_t *dstDevice, *src0Device, *src1Device;

    aclrtMallocHost((void**)(&dstHost), byteSize);
    aclrtMallocHost((void**)(&src0Host), byteSize);
    aclrtMallocHost((void**)(&src1Host), byteSize);

    aclrtMalloc((void**)&dstDevice, byteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&src0Device, byteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&src1Device, byteSize, ACL_MEM_MALLOC_HUGE_FIRST);

    uint8_t arr[64 * 64] = {};
    for (int i = 0; i < 64; i++) {
        for (int j = 0; j < 64; j++) {
            arr[i][j] = 1;
        }
    }
    src0Host = arr;

    uint8_t arr1[64 * 64] = {};
    for (int i = 0; i < 64; i++) {
        for (int j = 0; j < 64; j++) {
            arr1[i][j] = 2;
        }
    }
    src1Host = arr1;

    aclrtMemcpy(src0Device, byteSize, src0Host, byteSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(src1Device, byteSize, src1Host, byteSize, ACL_MEMCPY_HOST_TO_DEVICE);

    test_tadd<AS::UB, uint8_t, 16, 16, FractalLayoutKind::kRowMajor, 16, 16><<<16, nullptr, stream>>>(dstDevice,
                                                                                                      src0Device,
                                                                                                      src1Device);
    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, byteSize, dstDevice, byteSize, ACL_MEMCPY_DEVICE_TO_HOST);

    aclrtFree(dstDevice);
    aclrtFree(src0Device);
    aclrtFree(src1Device);

    aclrtFreeHost(dstHost);
    aclrtFreeHost(src0Host);
    aclrtFreeHost(src1Host);

    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();
    return 0;
}