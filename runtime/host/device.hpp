#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>
#include <sys/types.h>

namespace supernpu {

// One native host owns one resident gfrun session and its GM allocations.
// Submit one launch at a time. wait() makes all its device writes visible.
class Device {
public:
    Device(const std::string& gfrun, const std::string& firmware);
    ~Device();
    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;

    uint64_t allocate(size_t bytes, size_t alignment = 4096);
    void upload(uint64_t address, const void* data, size_t bytes);
    void download(void* data, uint64_t address, size_t bytes);
    void initialize();
    void launch(const std::string& kernel, unsigned blockDim, uint64_t argument);
    void wait();
    void close();

private:
    std::string receive();
    std::string response(const std::string& kind, uint64_t id);
    uint64_t command(const std::string& name, const std::string& args = {});
    void disconnect() noexcept;

    std::string firmware_;
    pid_t pid_ = -1;
    int socket_ = -1;
    FILE* input_ = nullptr;
    uint64_t sequence_ = 0;
    uint64_t pending_ = 0;
    uint32_t blocks_ = 0;
    bool initialized_ = false;
};
}
