#include "device.hpp"

#include <cerrno>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <fcntl.h>
#include <sstream>
#include <stdexcept>
#include <system_error>
#include <utility>
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

uint64_t kernelAddress(const std::string& image, const std::string& kernel) {
    std::ifstream symbols(image + ".kernels");
    uint64_t pc;
    std::string name;
    while (symbols >> std::hex >> pc >> name) {
        if (kernel == name) return pc;
    }
    throw std::runtime_error("missing device function " + kernel + " in " + image + ".kernels");
}
}

Device::Device(const std::string& gfrun, const std::string& firmware) : firmware_(firmware) {
    int pair[2];
    if (socketpair(AF_UNIX, SOCK_STREAM, 0, pair) < 0) systemFailure("socketpair");
    pid_ = fork();
    if (pid_ == 0) {
        ::close(pair[0]);
        if (dup2(pair[1], STDIN_FILENO) < 0 || dup2(pair[1], STDOUT_FILENO) < 0) _exit(126);
        ::close(pair[1]);
        execlp(gfrun.c_str(), gfrun.c_str(), "-f", firmware.c_str(), "--msgb",
               "--hscb-host", static_cast<char*>(nullptr));
        _exit(127);
    }
    ::close(pair[1]);
    socket_ = pair[0];
    try {
        if (pid_ < 0) systemFailure("fork");
        if (fcntl(socket_, F_SETFD, FD_CLOEXEC) < 0) systemFailure("fcntl");
#ifdef SO_NOSIGPIPE
        int on = 1;
        if (setsockopt(socket_, SOL_SOCKET, SO_NOSIGPIPE, &on, sizeof(on)) < 0)
            systemFailure("setsockopt");
#endif
        input_ = fdopen(socket_, "r");
        if (!input_) systemFailure("fdopen");
        const auto opened = receive();
        if (opened.compare(0, 5, "OPEN ") != 0) throw std::runtime_error("gfrun: " + opened);
    } catch (...) {
        disconnect();
        throw;
    }
}

void Device::disconnect() noexcept {
    if (socket_ >= 0) {
        shutdown(socket_, SHUT_RDWR);
        if (input_) fclose(input_);
        else ::close(socket_);
        input_ = nullptr;
        socket_ = -1;
    }
    if (pid_ > 0) {
        kill(pid_, SIGTERM);
        while (waitpid(pid_, nullptr, 0) < 0 && errno == EINTR) {}
        pid_ = -1;
    }
}
Device::~Device() { disconnect(); }

std::string Device::receive() {
    if (!input_) throw std::logic_error("device is closed");
    char* buffer = nullptr;
    size_t capacity = 0;
    while (getline(&buffer, &capacity, input_) >= 0) {
        std::string line(buffer);
        if (!line.empty() && line.back() == '\n') line.pop_back();
        if (line.compare(0, 10, "HSCB HOST ") == 0) {
            free(buffer);
            return line.substr(10);
        }
        // Keep scheduler events and simulator diagnostics visible in the run log.
        fprintf(stderr, "%s\n", line.c_str());
    }
    free(buffer);
    throw std::runtime_error("gfrun closed the HSCB channel before replying");
}

std::string Device::response(const std::string& kind, uint64_t id) {
    auto line = receive();
    std::istringstream fields(line);
    std::string actual;
    uint64_t received;
    if (!(fields >> actual >> received) || actual != kind || received != id)
        throw std::runtime_error("gfrun: " + line);
    return line;
}

uint64_t Device::command(const std::string& name, const std::string& args) {
    if (socket_ < 0 || pending_) throw std::logic_error("device closed or launch outstanding");
    const auto id = ++sequence_;
    const auto line = name + " " + std::to_string(id) + " " + args + "\n";
    size_t offset = 0;
    while (offset < line.size()) {
        int flags = 0;
#ifdef MSG_NOSIGNAL
        flags = MSG_NOSIGNAL;
#endif
        auto n = send(socket_, line.data() + offset, line.size() - offset, flags);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) systemFailure("send to gfrun");
        offset += static_cast<size_t>(n);
    }
    return id;
}

uint64_t Device::allocate(size_t bytes, size_t alignment) {
    auto id = command("ALLOC", std::to_string(bytes) + " " + std::to_string(alignment));
    auto line = response("ALLOC", id);
    if (field(line, "size") != bytes) throw std::runtime_error("short GM allocation");
    return field(line, "address");
}

void Device::initialize() {
    if (initialized_) throw std::logic_error("device already initialized");
    auto id = command("INIT");
    response("INITIALIZING", id);
    auto line = receive();
    if (line.compare(0, 6, "READY ") != 0 || !field(line, "workers")) throw std::runtime_error("no resident workers");
    initialized_ = true;
}

void Device::upload(uint64_t address, const void* data, size_t bytes) {
    if (!bytes) return;
    TransferFile file;
    std::ofstream out(file.path, std::ios::binary);
    out.write(static_cast<const char*>(data), bytes);
    out.close();
    if (!out) throw std::runtime_error("could not stage upload");
    std::ostringstream args;
    args << address << ' ' << file.path;
    auto id = command("WRITE", args.str());
    if (field(response("WRITE", id), "size") != bytes) throw std::runtime_error("short GM write");
}

void Device::download(void* data, uint64_t address, size_t bytes) {
    if (!bytes) return;
    TransferFile file;
    std::ostringstream args;
    args << address << ' ' << bytes << ' ' << file.path;
    auto id = command("READ", args.str());
    if (field(response("READ", id), "size") != bytes) throw std::runtime_error("short GM read");
    std::ifstream in(file.path, std::ios::binary);
    if (!in.read(static_cast<char*>(data), bytes)) throw std::runtime_error("short download");
}

void Device::launch(const std::string& kernel, unsigned blockDim, uint64_t argument) {
    if (!initialized_ || !blockDim) throw std::logic_error("launch needs initialized device and blockDim > 0");
    std::ostringstream args;
    args << kernelAddress(firmware_, kernel) << ' ' << argument << ' ' << blockDim;
    auto id = command("LAUNCH", args.str());
    auto line = response("SUBMITTED", id);
    if (field(line, "blocks") != blockDim) throw std::runtime_error("wrong submitted block count");
    pending_ = id;
    blocks_ = blockDim;
}

void Device::wait() {
    if (!pending_) throw std::logic_error("wait needs an outstanding launch");
    auto line = response("COMPLETE", pending_);
    pending_ = 0;
    if (field(line, "blocks") != blocks_ || field(line, "status") != 0)
        throw std::runtime_error("kernel failed: " + line);
}

void Device::close() {
    if (socket_ < 0) return;
    if (pending_) wait();
    auto id = command("QUIT");
    response("BYE", id);
    // Drain remaining diagnostics before closing the socket and joining gfrun.
    shutdown(socket_, SHUT_WR);
    char tail[512];
    while (fgets(tail, sizeof(tail), input_)) fputs(tail, stderr);
    int status;
    while (waitpid(pid_, &status, 0) < 0) {
        if (errno != EINTR) systemFailure("waitpid");
    }
    pid_ = -1;
    disconnect();
    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0)
        throw std::runtime_error("gfrun terminated unsuccessfully");
}
}
