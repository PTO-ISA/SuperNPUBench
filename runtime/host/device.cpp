#include "device.hpp"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <exception>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <functional>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <system_error>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>
#include <signal.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>

namespace supernpu {
namespace {

void systemFailure(const char* operation) {
    throw std::system_error(errno, std::generic_category(), operation);
}

struct TransferFile {
    std::string path;
    TransferFile() {
        auto name = (std::filesystem::temp_directory_path() / "hscb-transfer-XXXXXX").string();
        if (name.find_first_of("\r\n") != std::string::npos)
            throw std::runtime_error("temporary directory cannot be represented on the HSCB command line");
        int fd = mkstemp(name.data());
        if (fd < 0) systemFailure("mkstemp");
        ::close(fd);
        path = std::move(name);
    }
    ~TransferFile() { ::unlink(path.c_str()); }
};

uint64_t field(const std::string& line, const std::string& name) {
    std::istringstream fields(line);
    std::string token;
    const auto prefix = name + "=";
    while (fields >> token) {
        if (token.compare(0, prefix.size(), prefix) == 0) {
            const std::string value = token.substr(prefix.size());
            size_t used;
            auto number = std::stoull(value, &used);
            if (used == value.size()) return number;
        }
    }
    throw std::runtime_error("missing " + name + " in " + line);
}

uint64_t legacyKernelAddress(const std::string& image, const std::string& kernel) {
    std::ifstream symbols(image + ".kernels");
    uint64_t pc;
    std::string name;
    while (symbols >> std::hex >> pc >> name) {
        if (kernel == name) return pc;
    }
    throw std::runtime_error("missing device function " + kernel);
}

uint64_t nextOwner() {
    static std::atomic<uint64_t> owner{1};
    const auto value = owner.fetch_add(1, std::memory_order_relaxed);
    if (!value) throw std::overflow_error("device identity space exhausted");
    return value;
}

} // namespace

namespace detail {

struct Allocation {
    uint64_t address = 0;
    size_t size = 0;
    bool active = true;
};

class DeviceImpl {
public:
    struct Work {
        std::function<void()> action;
        std::vector<std::shared_ptr<Allocation>> dependencies;
    };
    struct DownloadCompletion {
        void* destination;
        std::shared_ptr<std::vector<unsigned char>> bytes;
    };

    DeviceImpl(const std::string& gfrun, std::string firmware)
        : owner(nextOwner()), firmware(std::move(firmware)) {
        int pair[2];
        if (socketpair(AF_UNIX, SOCK_STREAM, 0, pair) < 0) systemFailure("socketpair");
        pid = fork();
        if (pid == 0) {
            ::close(pair[0]);
            if (dup2(pair[1], STDIN_FILENO) < 0 || dup2(pair[1], STDOUT_FILENO) < 0) _exit(126);
            ::close(pair[1]);
            execlp(gfrun.c_str(), gfrun.c_str(), "-f", this->firmware.c_str(), "--msgb",
                   "--hscb-host", static_cast<char*>(nullptr));
            _exit(127);
        }
        ::close(pair[1]);
        socket = pair[0];
        try {
            if (pid < 0) systemFailure("fork");
            if (fcntl(socket, F_SETFD, FD_CLOEXEC) < 0) systemFailure("fcntl");
#ifdef SO_NOSIGPIPE
            int on = 1;
            if (setsockopt(socket, SOL_SOCKET, SO_NOSIGPIPE, &on, sizeof(on)) < 0)
                systemFailure("setsockopt");
#endif
            input = fdopen(socket, "r");
            if (!input) systemFailure("fdopen");
            receiveOpen();
        } catch (...) {
            disconnect();
            throw;
        }
    }

    ~DeviceImpl() { disconnect(); }

    void receiveOpen() {
        for (;;) {
            auto line = receiveLine();
            if (line.compare(0, 12, "HSCB KERNEL ") == 0) {
                announcedKernels = true;
                std::istringstream fields(line.substr(12));
                uint64_t pc;
                std::string name;
                if (!(fields >> std::hex >> pc >> name) || !pc || !(fields >> std::ws).eof())
                    throw std::runtime_error("gfrun: malformed kernel announcement");
                if (!kernels.emplace(name, pc).second)
                    throw std::runtime_error("gfrun: duplicate kernel announcement " + name);
                continue;
            }
            if (line.compare(0, 15, "HSCB HOST OPEN ") == 0) return;
            fprintf(stderr, "%s\n", line.c_str());
        }
    }

    std::string receiveLine() {
        if (!input) throw std::logic_error("device is closed");
        char* buffer = nullptr;
        size_t capacity = 0;
        if (getline(&buffer, &capacity, input) < 0) {
            free(buffer);
            throw std::runtime_error("gfrun closed the HSCB channel before replying");
        }
        std::string line(buffer);
        free(buffer);
        if (!line.empty() && line.back() == '\n') line.pop_back();
        return line;
    }

    std::string receive() {
        for (;;) {
            auto line = receiveLine();
            if (line.compare(0, 10, "HSCB HOST ") == 0) return line.substr(10);
            fprintf(stderr, "%s\n", line.c_str());
        }
    }

    std::string response(const std::string& kind, uint64_t id) {
        auto line = receive();
        std::istringstream fields(line);
        std::string actual;
        uint64_t received;
        if (!(fields >> actual >> received) || actual != kind || received != id)
            throw std::runtime_error("gfrun: " + line);
        if (line.find("status=ERROR") != std::string::npos)
            throw std::runtime_error("gfrun: " + line);
        return line;
    }

    uint64_t command(const std::string& name, const std::string& args = {}) {
        if (socket < 0) throw std::logic_error("device is closed");
        const auto id = ++sequence;
        const auto line = name + " " + std::to_string(id) + (args.empty() ? "" : " " + args) + "\n";
        size_t offset = 0;
        while (offset < line.size()) {
            int flags = 0;
#ifdef MSG_NOSIGNAL
            flags = MSG_NOSIGNAL;
#endif
            auto count = send(socket, line.data() + offset, line.size() - offset, flags);
            if (count < 0 && errno == EINTR) continue;
            if (count <= 0) systemFailure("send to gfrun");
            offset += static_cast<size_t>(count);
        }
        return id;
    }

    uint64_t directAllocate(size_t bytes, size_t alignment) {
        auto id = command("ALLOC", std::to_string(bytes) + " " + std::to_string(alignment));
        auto line = response("ALLOC", id);
        if (field(line, "size") != bytes) throw std::runtime_error("short GM allocation");
        return field(line, "address");
    }

    void directFree(uint64_t address) {
        auto id = command("FREE", std::to_string(address));
        response("FREE", id);
    }

    void directUpload(uint64_t address, const void* data, size_t bytes) {
        if (!bytes) return;
        TransferFile file;
        std::ofstream out(file.path, std::ios::binary);
        out.write(static_cast<const char*>(data), bytes);
        out.close();
        if (!out) throw std::runtime_error("could not stage upload");
        auto id = command("WRITE", std::to_string(address) + " " + file.path);
        if (field(response("WRITE", id), "size") != bytes)
            throw std::runtime_error("short GM write");
    }

    void directDownload(void* data, uint64_t address, size_t bytes) {
        if (!bytes) return;
        TransferFile file;
        auto id = command("READ", std::to_string(address) + " " + std::to_string(bytes) +
                                  " " + file.path);
        if (field(response("READ", id), "size") != bytes)
            throw std::runtime_error("short GM read");
        std::ifstream in(file.path, std::ios::binary);
        if (!in.read(static_cast<char*>(data), bytes)) throw std::runtime_error("short download");
    }

    void directLaunch(uint64_t pc, unsigned blockDim, uint64_t argument) {
        auto id = command("LAUNCH", std::to_string(pc) + " " + std::to_string(argument) +
                                   " " + std::to_string(blockDim));
        auto submitted = response("SUBMITTED", id);
        if (field(submitted, "blocks") != blockDim)
            throw std::runtime_error("wrong submitted block count");
        auto complete = response("COMPLETE", id);
        if (field(complete, "blocks") != blockDim || field(complete, "status") != 0)
            throw std::runtime_error("kernel failed: " + complete);
    }

    void startWorker() {
        worker = std::thread([this] { workerLoop(); });
    }

    void enqueue(Work work) {
        std::lock_guard<std::mutex> lock(mutex);
        if (streamError) std::rethrow_exception(streamError);
        if (stopping) throw std::logic_error("device stream is closed");
        queue.push_back(std::move(work));
        queueChanged.notify_one();
    }

    void workerLoop() noexcept {
        for (;;) {
            Work work;
            {
                std::unique_lock<std::mutex> lock(mutex);
                queueChanged.wait(lock, [this] { return stopping || !queue.empty(); });
                if (stopping && queue.empty()) return;
                work = std::move(queue.front());
                queue.pop_front();
                active = true;
            }
            try {
                work.action();
            } catch (...) {
                std::lock_guard<std::mutex> lock(mutex);
                if (!streamError) streamError = std::current_exception();
                queue.clear();
            }
            {
                std::lock_guard<std::mutex> lock(mutex);
                active = false;
                if (queue.empty()) idle.notify_all();
            }
        }
    }

    void synchronize() {
        if (!initialized) return;
        std::vector<DownloadCompletion> completedDownloads;
        std::unique_lock<std::mutex> lock(mutex);
        idle.wait(lock, [this] { return streamError || (!active && queue.empty()); });
        if (streamError) {
            downloads.clear();
            std::rethrow_exception(streamError);
        }
        completedDownloads.swap(downloads);
        lock.unlock();
        for (const auto& download : completedDownloads) {
            if (!download.bytes->empty())
                std::memcpy(download.destination, download.bytes->data(), download.bytes->size());
        }
    }

    void stopWorker() noexcept {
        if (!worker.joinable()) return;
        {
            std::lock_guard<std::mutex> lock(mutex);
            stopping = true;
            queueChanged.notify_all();
        }
        worker.join();
    }

    void abortWorker() noexcept {
        if (!worker.joinable()) return;
        {
            std::lock_guard<std::mutex> lock(mutex);
            stopping = true;
            queue.clear();
            downloads.clear();
            if (socket >= 0) shutdown(socket, SHUT_RDWR);
            queueChanged.notify_all();
        }
        worker.join();
    }

    void disconnect() noexcept {
        abortWorker();
        if (socket >= 0) {
            shutdown(socket, SHUT_RDWR);
            if (input) fclose(input);
            else ::close(socket);
            input = nullptr;
            socket = -1;
        }
        if (pid > 0) {
            kill(pid, SIGTERM);
            while (waitpid(pid, nullptr, 0) < 0 && errno == EINTR) {}
            pid = -1;
        }
    }

    void checkOpen() const {
        if (closed || socket < 0) throw std::logic_error("device is closed");
    }

    void checkOwner(uint64_t candidate) const {
        checkOpen();
        if (candidate != owner) throw std::invalid_argument("handle belongs to another device");
    }

    std::shared_ptr<Allocation> check(const Buffer& buffer, size_t offset = 0,
                                      size_t bytes = 0) const {
        checkOwner(buffer.owner_);
        if (!buffer.allocation_ || !buffer.allocation_->active)
            throw std::invalid_argument("buffer handle is stale");
        if (offset > buffer.allocation_->size || bytes > buffer.allocation_->size - offset)
            throw std::out_of_range("buffer range exceeds allocation");
        return buffer.allocation_;
    }

    uint64_t owner;
    std::string firmware;
    pid_t pid = -1;
    int socket = -1;
    FILE* input = nullptr;
    uint64_t sequence = 0;
    bool initialized = false;
    bool closed = false;
    bool announcedKernels = false;
    std::unordered_map<std::string, uint64_t> kernels;
    std::mutex mutex;
    std::condition_variable queueChanged;
    std::condition_variable idle;
    std::deque<Work> queue;
    std::vector<DownloadCompletion> downloads;
    std::thread worker;
    bool active = false;
    bool stopping = false;
    std::exception_ptr streamError;
};

} // namespace detail

Buffer::Buffer(uint64_t owner, std::shared_ptr<detail::Allocation> allocation)
    : owner_(owner), allocation_(std::move(allocation)) {}

size_t Buffer::size() const {
    return allocation_ && allocation_->active ? allocation_->size : 0;
}

Buffer::operator bool() const {
    return allocation_ && allocation_->active;
}

Device::Device(const std::string& gfrun, const std::string& firmware)
    : impl_(new detail::DeviceImpl(gfrun, firmware)) {}

Device::~Device() {
    if (impl_) impl_->disconnect();
}

Buffer Device::allocateBuffer(size_t bytes, size_t alignment) {
    impl_->checkOpen();
    if (!bytes) throw std::invalid_argument("buffer size must be nonzero");
    if (impl_->initialized) impl_->synchronize();
    const auto address = impl_->directAllocate(bytes, alignment);
    auto allocation = std::make_shared<detail::Allocation>();
    allocation->address = address;
    allocation->size = bytes;
    return Buffer(impl_->owner, std::move(allocation));
}

void Device::release(Buffer& buffer) {
    auto allocation = impl_->check(buffer);
    allocation->active = false;
    if (impl_->initialized) {
        impl_->enqueue({[this, allocation] { impl_->directFree(allocation->address); }, {}});
        impl_->synchronize();
    } else {
        impl_->directFree(allocation->address);
    }
    buffer = Buffer();
}

uint64_t Device::address(const Buffer& buffer) const {
    return impl_->check(buffer)->address;
}

Kernel Device::kernel(const std::string& name) const {
    impl_->checkOpen();
    const auto found = impl_->kernels.find(name);
    if (found == impl_->kernels.end())
        throw std::invalid_argument("kernel was not exported by the loaded image: " + name);
    return Kernel(impl_->owner, found->second, name);
}

void Device::enqueueUpload(const Buffer& destination, const void* source, size_t bytes,
                           size_t offset) {
    if (!impl_->initialized) throw std::logic_error("enqueueUpload requires initialize");
    if (!source && bytes) throw std::invalid_argument("upload source is null");
    auto allocation = impl_->check(destination, offset, bytes);
    std::vector<unsigned char> snapshot(bytes);
    if (bytes) std::memcpy(snapshot.data(), source, bytes);
    impl_->enqueue({[this, allocation, offset, snapshot = std::move(snapshot)] {
                        impl_->directUpload(allocation->address + offset,
                                            snapshot.data(), snapshot.size());
                    },
                    {allocation}});
}

void Device::enqueueDownload(void* destination, const Buffer& source, size_t bytes,
                             size_t offset) {
    if (!impl_->initialized) throw std::logic_error("enqueueDownload requires initialize");
    if (!destination && bytes) throw std::invalid_argument("download destination is null");
    auto allocation = impl_->check(source, offset, bytes);
    auto snapshot = std::make_shared<std::vector<unsigned char>>(bytes);
    {
        std::lock_guard<std::mutex> lock(impl_->mutex);
        if (impl_->streamError) std::rethrow_exception(impl_->streamError);
        if (impl_->stopping) throw std::logic_error("device stream is closed");
        impl_->downloads.push_back({destination, snapshot});
    }
    try {
        impl_->enqueue({[this, allocation, offset, snapshot] {
                        impl_->directDownload(snapshot->data(), allocation->address + offset,
                                              snapshot->size());
                    },
                    {allocation}});
    } catch (...) {
        std::lock_guard<std::mutex> lock(impl_->mutex);
        impl_->downloads.pop_back();
        throw;
    }
}

void Device::enqueueLaunchBytes(const Kernel& function, unsigned blockDim,
                                const void* args, size_t argsSize, size_t argsAlignment,
                                std::initializer_list<Buffer> dependencies) {
    impl_->checkOwner(function.owner_);
    if (!function.address_) throw std::invalid_argument("kernel handle is empty");
    if (!impl_->initialized || !blockDim)
        throw std::logic_error("launch needs initialized device and blockDim > 0");
    if (!args || !argsSize) throw std::invalid_argument("kernel arguments are empty");
    std::vector<std::shared_ptr<detail::Allocation>> retained;
    retained.reserve(dependencies.size());
    for (const auto& dependency : dependencies) retained.push_back(impl_->check(dependency));
    std::vector<unsigned char> snapshot(argsSize);
    std::memcpy(snapshot.data(), args, argsSize);
    const size_t alignment = std::max<size_t>(8, argsAlignment);
    impl_->enqueue({[this, function, blockDim, snapshot = std::move(snapshot), alignment] {
                        const uint64_t descriptor = impl_->directAllocate(snapshot.size(), alignment);
                        try {
                            impl_->directUpload(descriptor, snapshot.data(), snapshot.size());
                            impl_->directLaunch(function.address_, blockDim, descriptor);
                        } catch (...) {
                            try { impl_->directFree(descriptor); } catch (...) {}
                            throw;
                        }
                        impl_->directFree(descriptor);
                    },
                    std::move(retained)});
}

void Device::synchronize() { impl_->synchronize(); }

uint64_t Device::allocate(size_t bytes, size_t alignment) {
    impl_->checkOpen();
    if (impl_->initialized) impl_->synchronize();
    return impl_->directAllocate(bytes, alignment);
}

void Device::upload(uint64_t address, const void* data, size_t bytes) {
    impl_->checkOpen();
    if (impl_->initialized) {
        std::vector<unsigned char> snapshot(bytes);
        if (bytes) std::memcpy(snapshot.data(), data, bytes);
        impl_->enqueue({[this, address, snapshot = std::move(snapshot)] {
                            impl_->directUpload(address, snapshot.data(), snapshot.size());
                        }, {}});
        impl_->synchronize();
    } else {
        impl_->directUpload(address, data, bytes);
    }
}

void Device::download(void* data, uint64_t address, size_t bytes) {
    impl_->checkOpen();
    if (impl_->initialized) {
        impl_->enqueue({[this, data, address, bytes] {
                            impl_->directDownload(data, address, bytes);
                        }, {}});
        impl_->synchronize();
    } else {
        impl_->directDownload(data, address, bytes);
    }
}

void Device::initialize() {
    impl_->checkOpen();
    if (impl_->initialized) throw std::logic_error("device already initialized");
    auto id = impl_->command("INIT");
    impl_->response("INITIALIZING", id);
    auto line = impl_->receive();
    if (line.compare(0, 6, "READY ") != 0 || !field(line, "workers"))
        throw std::runtime_error("no resident workers");
    impl_->initialized = true;
    impl_->startWorker();
}

void Device::launch(const std::string& name, unsigned blockDim, uint64_t argument) {
    if (!impl_->initialized || !blockDim)
        throw std::logic_error("launch needs initialized device and blockDim > 0");
    uint64_t pc = 0;
    const auto found = impl_->kernels.find(name);
    if (found != impl_->kernels.end()) pc = found->second;
    else if (!impl_->announcedKernels) pc = legacyKernelAddress(impl_->firmware, name);
    else throw std::invalid_argument("kernel was not exported by the loaded image: " + name);
    impl_->enqueue({[this, pc, blockDim, argument] {
                        impl_->directLaunch(pc, blockDim, argument);
                    }, {}});
}

void Device::wait() { synchronize(); }

void Device::close() {
    if (!impl_ || impl_->closed) return;
    std::exception_ptr error;
    try { impl_->synchronize(); } catch (...) { error = std::current_exception(); }
    impl_->stopWorker();
    try {
        auto id = impl_->command("QUIT");
        impl_->response("BYE", id);
        shutdown(impl_->socket, SHUT_WR);
        char tail[512];
        while (fgets(tail, sizeof(tail), impl_->input)) fputs(tail, stderr);
        int status;
        while (waitpid(impl_->pid, &status, 0) < 0) {
            if (errno != EINTR) systemFailure("waitpid");
        }
        impl_->pid = -1;
        impl_->closed = true;
        impl_->disconnect();
        if (!WIFEXITED(status) || WEXITSTATUS(status) != 0)
            throw std::runtime_error("gfrun terminated unsuccessfully");
    } catch (...) {
        impl_->disconnect();
        if (!error) error = std::current_exception();
    }
    if (error) std::rethrow_exception(error);
}

} // namespace supernpu
