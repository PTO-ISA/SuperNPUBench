#include "device.hpp"
#include "tadd_args.hpp"
#include "fill_args.hpp"

#include <array>
#include <iostream>
#include <stdexcept>

namespace {

template <std::size_t Size>
void checkGeometry(const std::array<uint64_t, Size>& observed,
                   unsigned blockDim, const char* launch) {
    for (unsigned block = 0; block < observed.size(); ++block) {
        const uint64_t expected = block < blockDim
                                      ? (uint64_t(blockDim) << 32) | block
                                      : UINT64_MAX;
        if (observed[block] != expected) {
            throw std::runtime_error(std::string(launch) +
                                     " launch has incorrect block geometry");
        }
    }
}

}  // namespace

int main(int argc, char** argv) try {
    if (argc != 3) throw std::runtime_error("usage: tadd_host GFRUN WORKERS.ELF");

    supernpu::Device device(argv[1], argv[2]);
    constexpr unsigned tiles = 17;
    constexpr unsigned elements = tiles * kTileElements;
    constexpr unsigned firstBlockDim = 8;
    constexpr unsigned secondBlockDim = 4;

    std::array<float, elements> a, b, c, result;
    std::array<uint64_t, firstBlockDim> firstGeometry;
    std::array<uint64_t, firstBlockDim> secondGeometry;
    std::array<int32_t, 37> zeroResult;
    for (unsigned element = 0; element < elements; ++element) {
        a[element] = static_cast<int>(element % 97) - 48;
        b[element] = static_cast<int>(element % 31) - 15;
        c[element] = static_cast<int>(element % 17) - 8;
    }
    result.fill(-98765.0f);
    firstGeometry.fill(UINT64_MAX);
    secondGeometry.fill(UINT64_MAX);
    zeroResult.fill(-1);

    auto aDevice = device.allocateBuffer(sizeof(a));
    auto bDevice = device.allocateBuffer(sizeof(b));
    auto cDevice = device.allocateBuffer(sizeof(c));
    auto temporary = device.allocateBuffer(sizeof(result));
    auto output = device.allocateBuffer(sizeof(result));
    auto firstGeometryDevice = device.allocateBuffer(sizeof(firstGeometry));
    auto secondGeometryDevice = device.allocateBuffer(sizeof(secondGeometry));
    auto zeroDevice = device.allocateBuffer(sizeof(zeroResult));
    const uint64_t temporaryAddress = device.address(temporary);
    const auto staleTemporary = temporary;

    device.initialize();
    const auto tadd = device.kernel("bench_tadd");
    const auto fill = device.kernel("bench_fill");

    device.enqueueUpload(aDevice, a.data(), sizeof(a));
    device.enqueueUpload(bDevice, b.data(), sizeof(b));
    device.enqueueUpload(cDevice, c.data(), sizeof(c));
    device.enqueueUpload(firstGeometryDevice, firstGeometry.data(),
                         sizeof(firstGeometry));
    device.enqueueUpload(secondGeometryDevice, secondGeometry.data(),
                         sizeof(secondGeometry));
    device.enqueueUpload(zeroDevice, zeroResult.data(), sizeof(zeroResult));

    supernpu::KernelCall<kTaddKernelSchema> args;
    args.buffer(aDevice)
        .buffer(bDevice)
        .buffer(temporary)
        .buffer(firstGeometryDevice)
        .u64(tiles);
    device.enqueueLaunch(tadd, firstBlockDim, args);

    // Reuse and mutate the same host object without waiting for the first
    // launch to finish. The stream must preserve the first argument snapshot
    // while taking a new snapshot for the dependent second launch.
    args = {};
    args.buffer(temporary)
        .buffer(cDevice)
        .buffer(output)
        .buffer(secondGeometryDevice)
        .u64(tiles);
    device.enqueueLaunch(tadd, secondBlockDim, args);

    supernpu::KernelCall<kFillKernelSchema> fillZero;
    fillZero.buffer(zeroDevice)
        .u32(static_cast<uint32_t>(zeroResult.size()))
        .i32(-3)
        .u64(0);
    device.enqueueLaunch(fill, 3, fillZero);

    device.enqueueDownload(result.data(), output, sizeof(result));
    device.enqueueDownload(firstGeometry.data(), firstGeometryDevice,
                           sizeof(firstGeometry));
    device.enqueueDownload(secondGeometry.data(), secondGeometryDevice,
                           sizeof(secondGeometry));
    device.enqueueDownload(zeroResult.data(), zeroDevice, sizeof(zeroResult));
    device.synchronize();

    for (unsigned element = 0; element < elements; ++element) {
        const float expected = a[element] + b[element] + c[element];
        if (result[element] != expected) {
            std::cerr << "element " << element << " expected " << expected
                      << " got " << result[element] << '\n';
            return 1;
        }
    }
    checkGeometry(firstGeometry, firstBlockDim, "first");
    checkGeometry(secondGeometry, secondBlockDim, "second");
    for (const auto value : zeroResult) {
        if (value != -3)
            throw std::runtime_error("fill kernel arguments were not preserved");
    }

    device.release(temporary);
    auto reused = device.allocateBuffer(sizeof(result));
    if (device.address(reused) != temporaryAddress)
        throw std::runtime_error("released device allocation was not reused");

    // The old handle must remain stale even after its numeric address is
    // assigned to a new allocation generation.
    bool rejectedStaleHandle = false;
    try {
        (void)device.address(staleTemporary);
    } catch (const std::logic_error&) {
        rejectedStaleHandle = true;
    }
    if (!rejectedStaleHandle)
        throw std::runtime_error("released buffer handle remained usable");

    device.release(reused);
    device.release(secondGeometryDevice);
    device.release(firstGeometryDevice);
    device.release(output);
    device.release(cDevice);
    device.release(bDevice);
    device.release(aDevice);
    device.release(zeroDevice);
    device.close();

    std::cout << "PASS default stream: TADD(a,b)->tmp, TADD(tmp,c)->out, "
              << elements << " FP32 results equal a+b+c\n"
              << "PASS launch snapshots: blockDim=8 then blockDim=4, "
              << "independent geometry and mutable host args preserved\n"
              << "PASS device buffer: temporary stayed in GM, address "
              << temporaryAddress << " was freed and reused, stale handle rejected\n";
    std::cout << "PASS generic launch: two kernel entries, typed buffer/u64/u32/i32 "
                 "arguments, signed value and zero scalar verified\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
