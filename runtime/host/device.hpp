#pragma once

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>

namespace supernpu {

namespace detail {
struct Allocation;
class DeviceImpl;
}

class Buffer {
public:
    Buffer() = default;

    size_t size() const;
    explicit operator bool() const;

private:
    friend class Device;
    friend class detail::DeviceImpl;
    Buffer(uint64_t owner, std::shared_ptr<detail::Allocation> allocation);

    uint64_t owner_ = 0;
    std::shared_ptr<detail::Allocation> allocation_;
};

class Kernel {
public:
    Kernel() = default;
    explicit operator bool() const { return address_ != 0; }
    const std::string& name() const { return name_; }

private:
    friend class Device;
    Kernel(uint64_t owner, uint64_t address, std::string name)
        : owner_(owner), address_(address), name_(std::move(name)) {}

    uint64_t owner_ = 0;
    uint64_t address_ = 0;
    std::string name_;
};

// One host thread submits to one resident gfrun session and one FIFO default
// stream. enqueue* calls return after copying host input into runtime-owned
// storage. A download destination must remain valid until synchronize(). The
// destructor cancels queued work and never publishes pending downloads.
class Device {
public:
    Device(const std::string& gfrun, const std::string& firmware);
    ~Device();
    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;

    Buffer allocateBuffer(size_t bytes, size_t alignment = 4096);
    void release(Buffer& buffer);
    uint64_t address(const Buffer& buffer) const;
    Kernel kernel(const std::string& name) const;

    void enqueueUpload(const Buffer& destination, const void* source, size_t bytes,
                       size_t offset = 0);
    void enqueueDownload(void* destination, const Buffer& source, size_t bytes,
                         size_t offset = 0);

    template <class Args>
    void enqueueLaunch(const Kernel& function, unsigned blockDim, const Args& args,
                       std::initializer_list<Buffer> dependencies) {
        static_assert(std::is_trivially_copyable<Args>::value,
                      "kernel arguments must be trivially copyable");
        enqueueLaunchBytes(function, blockDim, &args, sizeof(args), alignof(Args),
                           dependencies);
    }

    void synchronize();

    // Compatibility entry points. Transfers are synchronous. launch() submits
    // to the default stream and wait() synchronizes it.
    uint64_t allocate(size_t bytes, size_t alignment = 4096);
    void upload(uint64_t address, const void* data, size_t bytes);
    void download(void* data, uint64_t address, size_t bytes);
    void initialize();
    void launch(const std::string& kernel, unsigned blockDim, uint64_t argument);
    void wait();
    void close();

private:
    void enqueueLaunchBytes(const Kernel& function, unsigned blockDim,
                            const void* args, size_t argsSize, size_t argsAlignment,
                            std::initializer_list<Buffer> dependencies);

    std::unique_ptr<detail::DeviceImpl> impl_;
};

}
