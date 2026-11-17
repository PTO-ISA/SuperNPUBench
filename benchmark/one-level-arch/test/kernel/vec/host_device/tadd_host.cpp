#include "device.hpp"
#include "tadd_args.hpp"

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
    for (unsigned element = 0; element < elements; ++element) {
        a[element] = static_cast<int>(element % 97) - 48;
        b[element] = static_cast<int>(element % 31) - 15;
        c[element] = static_cast<int>(element % 17) - 8;
    }
    result.fill(-98765.0f);
    firstGeometry.fill(UINT64_MAX);
    secondGeometry.fill(UINT64_MAX);

    auto aDevice = device.allocateBuffer(sizeof(a));
    auto bDevice = device.allocateBuffer(sizeof(b));
    auto cDevice = device.allocateBuffer(sizeof(c));
    auto temporary = device.allocateBuffer(sizeof(result));
    auto output = device.allocateBuffer(sizeof(result));
    auto firstGeometryDevice = device.allocateBuffer(sizeof(firstGeometry));
    auto secondGeometryDevice = device.allocateBuffer(sizeof(secondGeometry));
    const uint64_t temporaryAddress = device.address(temporary);
    const auto staleTemporary = temporary;

    device.initialize();
    const auto tadd = device.kernel("bench_tadd");

    device.enqueueUpload(aDevice, a.data(), sizeof(a));
    device.enqueueUpload(bDevice, b.data(), sizeof(b));
    device.enqueueUpload(cDevice, c.data(), sizeof(c));
    device.enqueueUpload(firstGeometryDevice, firstGeometry.data(),
                         sizeof(firstGeometry));
    device.enqueueUpload(secondGeometryDevice, secondGeometry.data(),
                         sizeof(secondGeometry));

    TaddArgs args{device.address(aDevice), device.address(bDevice),
                  device.address(temporary), device.address(firstGeometryDevice),
                  tiles};
    device.enqueueLaunch(tadd, firstBlockDim, args,
                         {aDevice, bDevice, temporary, firstGeometryDevice});

    // Reuse and mutate the same host object without waiting for the first
    // launch to finish. The stream must preserve the first argument snapshot
    // while taking a new snapshot for the dependent second launch.
    args = TaddArgs{device.address(temporary), device.address(cDevice),
                    device.address(output), device.address(secondGeometryDevice),
                    tiles};
    device.enqueueLaunch(tadd, secondBlockDim, args,
                         {temporary, cDevice, output, secondGeometryDevice});

    device.enqueueDownload(result.data(), output, sizeof(result));
    device.enqueueDownload(firstGeometry.data(), firstGeometryDevice,
                           sizeof(firstGeometry));
    device.enqueueDownload(secondGeometry.data(), secondGeometryDevice,
                           sizeof(secondGeometry));
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
    device.close();

    std::cout << "PASS default stream: TADD(a,b)->tmp, TADD(tmp,c)->out, "
              << elements << " FP32 results equal a+b+c\n"
              << "PASS launch snapshots: blockDim=8 then blockDim=4, "
              << "independent geometry and mutable host args preserved\n"
              << "PASS device buffer: temporary stayed in GM, address "
              << temporaryAddress << " was freed and reused, stale handle rejected\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
