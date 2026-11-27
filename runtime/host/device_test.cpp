#include "device.hpp"

#include <cassert>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>
#include <unistd.h>

struct Args {
    uint64_t value;
};

int main(int argc, char** argv) {
    assert(argc == 2);
    const char* fake = argv[1];

    {
        supernpu::Device first(fake, "unused.elf");
        supernpu::Device second(fake, "unused.elf");
        auto buffer = first.allocateBuffer(64);
        const auto address = first.address(buffer);
        bool crossSessionRejected = false;
        try {
            (void)second.address(buffer);
        } catch (const std::invalid_argument&) {
            crossSessionRejected = true;
        }
        assert(crossSessionRejected);

        bool preinitializeQueueRejected = false;
        try {
            first.enqueueUpload(buffer, "x", 1);
        } catch (const std::logic_error&) {
            preinitializeQueueRejected = true;
        }
        assert(preinitializeQueueRejected);

        const auto stale = buffer;
        first.release(buffer);
        auto reused = first.allocateBuffer(64);
        assert(first.address(reused) == address);
        assert(!stale);
        bool staleRejected = false;
        try {
            (void)first.address(stale);
        } catch (const std::invalid_argument&) {
            staleRejected = true;
        }
        assert(staleRejected);
        first.release(reused);
        first.close();
        second.close();
    }

    {
        const std::string marker = "/tmp/supernpu-host-runtime-" +
                                   std::to_string(static_cast<long long>(getpid()));
        supernpu::Device stream(fake, marker);
        auto buffer = stream.allocateBuffer(64);
        stream.initialize();
        const auto function = stream.kernel("ok_kernel");
        bool rangeRejected = false;
        try {
            stream.enqueueUpload(buffer, "xx", 2, 63);
        } catch (const std::out_of_range&) {
            rangeRejected = true;
        }
        assert(rangeRejected);
        stream.enqueueLaunch(function, 1, Args{1}, {buffer});
        supernpu::KernelCall<0x54455354u> generic;
        generic.buffer(buffer).i32(0);
        stream.enqueueLaunch(function, 1, generic);
        std::ofstream(marker + ".submitted").put('\n');
        stream.synchronize();
        std::remove((marker + ".submitted").c_str());
        stream.release(buffer);
        bool dependencyRejected = false;
        try {
            stream.enqueueLaunch(function, 1, Args{3}, {buffer});
        } catch (const std::invalid_argument&) {
            dependencyRejected = true;
        }
        assert(dependencyRejected);
        stream.close();
    }

    {
        supernpu::Device failing(fake, "unused.elf");
        auto buffer = failing.allocateBuffer(64);
        failing.initialize();
        failing.enqueueLaunch(failing.kernel("fail_kernel"), 1, Args{1}, {buffer});
        bool synchronizeWoke = false;
        try {
            failing.synchronize();
        } catch (const std::runtime_error&) {
            synchronizeWoke = true;
        }
        assert(synchronizeWoke);
    }
}
