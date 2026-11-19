#include "device.hpp"
#include "fill_args.hpp"

#include <iostream>
#include <stdexcept>
#include <string>

int main(int argc, char** argv) try {
    if (argc != 3)
        throw std::runtime_error("usage: invalid_descriptor_host GFRUN WORKERS.ELF");
    supernpu::Device device(argv[1], argv[2]);
    device.initialize();

    const supernpu::KernelArgumentHeader invalid{
        supernpu::kKernelArgumentMagic, supernpu::kKernelArgumentVersion,
        sizeof(supernpu::KernelArgumentHeader),
        4 * sizeof(supernpu::KernelArgumentRecord), kFillKernelSchema ^ 1u};
    device.enqueueLaunch(device.kernel("bench_fill"), 1, invalid, {});
    try {
        device.synchronize();
    } catch (const std::runtime_error& error) {
        if (std::string(error.what()).find("status=262") == std::string::npos)
            throw;
        std::cout << "PASS generic launch rejects a mismatched device schema\n";
        return 0;
    }
    throw std::runtime_error("invalid descriptor unexpectedly completed");
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
