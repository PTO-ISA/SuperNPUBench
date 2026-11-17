#include "device.hpp"
#include "tadd_args.hpp"

#include <array>
#include <iostream>
#include <stdexcept>

int main(int argc, char** argv) try {
    if (argc != 3) throw std::runtime_error("usage: tadd_host GFRUN WORKERS.ELF");
    supernpu::Device device(argv[1], argv[2]);
    constexpr unsigned tiles = 17;
    std::array<float, tiles * kTileElements> a, b, result;
    std::array<uint64_t, 8> geometry;
    TaddArgs args{device.allocate(sizeof(a)), device.allocate(sizeof(b)),
                  device.allocate(sizeof(result)), device.allocate(sizeof(geometry)), tiles};
    const auto descriptor = device.allocate(sizeof(args));
    device.upload(descriptor, &args, sizeof(args));
    for (unsigned i = 0; i < a.size(); ++i) {
        a[i] = static_cast<int>(i % 97) - 48;
        b[i] = static_cast<int>(i % 31) - 15;
    }
    device.upload(args.a, a.data(), sizeof(a));
    device.upload(args.b, b.data(), sizeof(b));
    device.initialize();
    for (const unsigned blocks : {8u, 4u}) {
        if (blocks == 4) {
            // Derive the second input from the completed first launch.
            a = result;
            device.upload(args.a, a.data(), sizeof(a));
        }
        result.fill(-98765.0f);
        geometry.fill(UINT64_MAX);
        device.upload(args.out, result.data(), sizeof(result));
        device.upload(args.geometry, geometry.data(), sizeof(geometry));
        device.launch("bench_tadd", blocks, descriptor);
        device.wait();
        device.download(result.data(), args.out, sizeof(result));
        device.download(geometry.data(), args.geometry, sizeof(geometry));
        for (unsigned i = 0; i < result.size(); ++i) {
            if (result[i] != a[i] + b[i]) {
                std::cerr << "element " << i << " expected " << a[i] + b[i]
                          << " got " << result[i] << '\n';
                return 1;
            }
        }
        for (unsigned i = 0; i < geometry.size(); ++i) {
            const uint64_t expected = i < blocks ? (uint64_t(blocks) << 32) | i : UINT64_MAX;
            if (geometry[i] != expected) throw std::runtime_error("unexpected logical block geometry");
        }
        std::cout << "PASS blockDim=" << blocks << ", " << result.size()
                  << " TADD elements, 8 geometry slots checked\n";
    }
    device.close();
    return 0;
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
