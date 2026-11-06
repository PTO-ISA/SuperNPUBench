#include "communication/mapped_window.hpp"

#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>
#include <thread>

#include <signal.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>

namespace {

using supernpu::comm::Status;
constexpr std::size_t kWidth = 32;
constexpr std::size_t kCapacity = 96;
constexpr float kSentinel = -8192.0f;

struct CpuPayload {
    static constexpr std::size_t kWidth = ::kWidth;
    using Tile = std::array<float, kWidth>;

    static void load(Tile& tile, const float* source) {
        std::memcpy(tile.data(), source, sizeof(tile));
    }
    static void store(float* destination, Tile& tile) {
        std::memcpy(destination, tile.data(), sizeof(tile));
    }
    static void zero(Tile& tile) { tile.fill(0.0f); }
    static void scale(Tile& destination, Tile& source, float weight) {
        for (std::size_t i = 0; i < kWidth; ++i) destination[i] = source[i] * weight;
    }
    static void add(Tile& destination, Tile& lhs, Tile& rhs) {
        for (std::size_t i = 0; i < kWidth; ++i) destination[i] = lhs[i] + rhs[i];
    }
};

static_assert(std::atomic_ref<std::uint64_t>::is_always_lock_free,
              "The MMAP protocol requires native lock-free uint64 atomics");

class HostSync {
public:
    explicit HostSync(std::chrono::milliseconds timeout = std::chrono::seconds(2))
        : timeout_(timeout) {}

    bool wait_equal(std::uint64_t& word, std::uint64_t expected) {
        std::atomic_ref<std::uint64_t> value(word);
        if (!value.is_lock_free()) return false;
        const auto deadline = std::chrono::steady_clock::now() + timeout_;
        std::uint32_t spins = 0;
        while (value.load(std::memory_order_acquire) != expected) {
            if (std::chrono::steady_clock::now() >= deadline) return false;
            if ((++spins & 255U) == 0U)
                std::this_thread::sleep_for(std::chrono::microseconds(20));
            else
                std::this_thread::yield();
        }
        return true;
    }

    void publish(std::uint64_t& word, std::uint64_t epoch) {
        std::atomic_ref<std::uint64_t> value(word);
        value.store(epoch, std::memory_order_release);
    }

private:
    std::chrono::milliseconds timeout_;
};

using Window = supernpu::comm::Window<kCapacity>;

struct SharedState {
    Window single[2];
    Window combine[3][2];
};

class Mapping {
public:
    Mapping() {
        address_ = mmap(nullptr, sizeof(SharedState), PROT_READ | PROT_WRITE,
                        MAP_SHARED | MAP_ANONYMOUS, -1, 0);
        if (address_ != MAP_FAILED) new (address_) SharedState{};
    }
    ~Mapping() {
        if (address_ != MAP_FAILED) {
            static_cast<SharedState*>(address_)->~SharedState();
            munmap(address_, sizeof(SharedState));
        }
    }
    Mapping(const Mapping&) = delete;
    Mapping& operator=(const Mapping&) = delete;
    bool valid() const { return address_ != MAP_FAILED; }
    SharedState* get() { return static_cast<SharedState*>(address_); }

private:
    void* address_ = MAP_FAILED;
};

struct TestContext {
    int failures = 0;

    void expect(bool condition, const char* message) {
        if (!condition) {
            std::fprintf(stderr, "FAIL: %s\n", message);
            ++failures;
        }
    }
};

std::size_t elements_for(std::size_t iteration) {
    return kWidth * (1 + iteration % 3);
}

void fill_payload(float* data, std::size_t elements, int producer, std::size_t iteration) {
    for (std::size_t i = 0; i < kCapacity; ++i) data[i] = kSentinel;
    for (std::size_t i = 0; i < elements; ++i)
        data[i] = static_cast<float>(producer * 128 + static_cast<int>(iteration) * 4 +
                                     static_cast<int>(i % 4));
}

bool payload_matches(const float* actual, const float* expected, std::size_t elements) {
    for (std::size_t i = 0; i < elements; ++i)
        if (actual[i] != expected[i]) return false;
    for (std::size_t i = elements; i < kCapacity; ++i)
        if (actual[i] != kSentinel) return false;
    return true;
}

bool wait_children(const pid_t* children, std::size_t count,
                   std::chrono::seconds timeout, TestContext& test) {
    std::array<bool, 3> reaped{};
    std::size_t remaining = count;
    bool ok = true;
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    while (remaining != 0 && std::chrono::steady_clock::now() < deadline) {
        for (std::size_t i = 0; i < count; ++i) {
            if (reaped[i]) continue;
            int status = 0;
            const pid_t result = waitpid(children[i], &status, WNOHANG);
            if (result == children[i]) {
                reaped[i] = true;
                --remaining;
                if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) ok = false;
            } else if (result < 0 && errno != EINTR) {
                reaped[i] = true;
                --remaining;
                ok = false;
            }
        }
        if (remaining != 0) std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    if (remaining != 0) {
        ok = false;
        for (std::size_t i = 0; i < count; ++i)
            if (!reaped[i]) kill(children[i], SIGKILL);
        for (std::size_t i = 0; i < count; ++i) {
            if (reaped[i]) continue;
            int status = 0;
            while (waitpid(children[i], &status, 0) < 0 && errno == EINTR) {}
        }
    }
    test.expect(ok, "child completion and exit status");
    return ok;
}

void terminate_children(const pid_t* children, std::size_t count) {
    for (std::size_t i = 0; i < count; ++i) {
        int status = 0;
        pid_t result;
        do {
            result = waitpid(children[i], &status, WNOHANG);
        } while (result < 0 && errno == EINTR);
        if (result == children[i] || (result < 0 && errno == ECHILD)) continue;
        if (result == 0) {
            kill(children[i], SIGKILL);
            do {
                result = waitpid(children[i], &status, 0);
            } while (result < 0 && errno == EINTR);
        }
    }
}

void test_direct_operations(TestContext& test) {
    std::array<float, kCapacity> source{}, copy{}, result{};
    fill_payload(source.data(), kCapacity, 1, 2);
    copy.fill(kSentinel);
    test.expect(supernpu::comm::put<CpuPayload>(copy.data(), source.data(), kCapacity) == Status::ok,
                "put accepts a full tile-aligned range");
    test.expect(copy == source, "put copies every element");
    result.fill(kSentinel);
    test.expect(supernpu::comm::get<CpuPayload>(result.data(), copy.data(), kCapacity) == Status::ok,
                "get accepts a full tile-aligned range");
    test.expect(result == source, "get copies every element");

    std::array<float, kCapacity> second{};
    fill_payload(second.data(), kCapacity, 2, 1);
    const float* sources[] = {source.data(), second.data()};
    const float weights[] = {0.5f, -0.25f};
    test.expect(supernpu::comm::weighted_combine<CpuPayload>(
                    result.data(), sources, weights, 2, kCapacity) == Status::ok,
                "weighted combine succeeds");
    bool weighted_ok = true;
    for (std::size_t i = 0; i < kCapacity; ++i)
        weighted_ok &= result[i] == source[i] * 0.5f - second[i] * 0.25f;
    test.expect(weighted_ok, "weighted combine matches scalar golden");
    test.expect(supernpu::comm::reduce_sum<CpuPayload>(
                    result.data(), sources, 2, kCapacity) == Status::ok,
                "reduce sum succeeds");
    bool sum_ok = true;
    for (std::size_t i = 0; i < kCapacity; ++i)
        sum_ok &= result[i] == source[i] + second[i];
    test.expect(sum_ok, "reduce sum matches scalar golden");

    test.expect(supernpu::comm::put<CpuPayload>(nullptr, source.data(), kWidth) ==
                    Status::invalid_argument,
                "put rejects null destination");
    test.expect(supernpu::comm::get<CpuPayload>(result.data(), nullptr, kWidth) ==
                    Status::invalid_argument,
                "get rejects null source");
    test.expect(supernpu::comm::put<CpuPayload>(copy.data(), source.data(), kWidth + 1) ==
                    Status::invalid_argument,
                "put rejects a partial tile");
    test.expect(supernpu::comm::weighted_combine<CpuPayload>(
                    result.data(), sources, weights, 0, kWidth) == Status::invalid_argument,
                "combine rejects zero sources");
    test.expect(supernpu::comm::weighted_combine<CpuPayload>(
                    result.data(), sources, weights, supernpu::comm::kMaxSources + 1,
                    kWidth) == Status::invalid_argument,
                "combine rejects too many sources before dereferencing them");
}

bool run_single_producer(SharedState* shared, std::size_t iterations) {
    HostSync sync(std::chrono::seconds(2));
    std::array<float, kCapacity> source{};
    for (std::size_t iteration = 0; iteration < iterations; ++iteration) {
        const std::size_t slot = iteration % 2;
        const std::uint64_t epoch = iteration / 2 + 1;
        const std::size_t elements = elements_for(iteration);
        fill_payload(source.data(), elements, 1, iteration);
        if ((iteration % 5) == 0) std::this_thread::sleep_for(std::chrono::microseconds(80));
        if (supernpu::comm::send<CpuPayload>(shared->single[slot], source.data(),
                                             elements, epoch, sync) != Status::ok)
            return false;
    }
    return true;
}

void test_single_producer_protocol(TestContext& test) {
    Mapping mapping;
    test.expect(mapping.valid(), "MAP_SHARED allocation for one producer");
    if (!mapping.valid()) return;
    constexpr std::size_t kIterations = 36;
    const pid_t child = fork();
    if (child == 0) _exit(run_single_producer(mapping.get(), kIterations) ? 0 : 10);
    test.expect(child > 0, "fork one producer");
    if (child <= 0) return;

    HostSync sync(std::chrono::seconds(2));
    std::array<float, kCapacity> output{}, golden{};
    bool protocol_ok = true;
    const auto scenario_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(12);
    for (std::size_t iteration = 0; iteration < kIterations; ++iteration) {
        if (std::chrono::steady_clock::now() >= scenario_deadline) {
            protocol_ok = false;
            break;
        }
        const std::size_t elements = elements_for(iteration);
        output.fill(kSentinel);
        fill_payload(golden.data(), elements, 1, iteration);
        if ((iteration % 7) == 0) std::this_thread::sleep_for(std::chrono::microseconds(120));
        const Status status = supernpu::comm::receive<CpuPayload>(
            output.data(), mapping.get()->single[iteration % 2], elements,
            iteration / 2 + 1, sync);
        if (status != Status::ok || !payload_matches(output.data(), golden.data(), elements)) {
            protocol_ok = false;
            break;
        }
    }
    test.expect(protocol_ok, "two-rank ping-pong Put/Get protocol and sentinels");
    const pid_t children[] = {child};
    if (protocol_ok)
        wait_children(children, 1, std::chrono::seconds(8), test);
    else
        terminate_children(children, 1);
}

bool run_combine_producer(SharedState* shared, int producer, std::size_t iterations) {
    HostSync sync(std::chrono::seconds(2));
    std::array<float, kCapacity> source{};
    for (std::size_t iteration = 0; iteration < iterations; ++iteration) {
        const std::size_t elements = elements_for(iteration);
        fill_payload(source.data(), elements, producer + 1, iteration);
        if (((iteration + static_cast<std::size_t>(producer)) % 4) == 0)
            std::this_thread::sleep_for(std::chrono::microseconds(50 + producer * 40));
        if (supernpu::comm::send<CpuPayload>(shared->combine[producer][iteration % 2],
                                             source.data(), elements, iteration / 2 + 1,
                                             sync) != Status::ok)
            return false;
    }
    return true;
}

void test_four_rank_combine(TestContext& test) {
    Mapping mapping;
    test.expect(mapping.valid(), "MAP_SHARED allocation for three producers");
    if (!mapping.valid()) return;
    constexpr std::size_t kIterations = 30;
    pid_t children[3]{};
    std::size_t started = 0;
    for (int producer = 0; producer < 3; ++producer) {
        const pid_t child = fork();
        if (child == 0) _exit(run_combine_producer(mapping.get(), producer, kIterations) ? 0 : 20 + producer);
        if (child < 0) break;
        children[started++] = child;
    }
    test.expect(started == 3, "fork three producers");
    if (started != 3) {
        terminate_children(children, started);
        return;
    }

    HostSync sync(std::chrono::seconds(2));
    const float weights[] = {0.5f, -0.25f, 2.0f};
    std::array<float, kCapacity> output{};
    bool protocol_ok = true;
    const auto scenario_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(12);
    for (std::size_t iteration = 0; iteration < kIterations; ++iteration) {
        if (std::chrono::steady_clock::now() >= scenario_deadline) {
            protocol_ok = false;
            break;
        }
        const std::size_t elements = elements_for(iteration);
        const std::size_t slot = iteration % 2;
        Window* windows[] = {&mapping.get()->combine[0][slot],
                             &mapping.get()->combine[1][slot],
                             &mapping.get()->combine[2][slot]};
        output.fill(kSentinel);
        const float* selected_weights = (iteration % 2 == 0) ? weights : nullptr;
        const Status status = supernpu::comm::receive_combine<CpuPayload>(
            output.data(), windows, selected_weights, 3, elements, iteration / 2 + 1, sync);
        if (status != Status::ok) {
            protocol_ok = false;
            break;
        }
        bool iteration_ok = true;
        for (std::size_t i = 0; i < elements; ++i) {
            float golden = 0.0f;
            for (int producer = 0; producer < 3; ++producer) {
                const float value = static_cast<float>((producer + 1) * 128 +
                    static_cast<int>(iteration) * 4 + static_cast<int>(i % 4));
                golden += value * (selected_weights ? selected_weights[producer] : 1.0f);
            }
            iteration_ok &= output[i] == golden;
        }
        for (std::size_t i = elements; i < kCapacity; ++i)
            iteration_ok &= output[i] == kSentinel;
        if (!iteration_ok) {
            protocol_ok = false;
            break;
        }
        if ((iteration % 6) == 0) std::this_thread::sleep_for(std::chrono::microseconds(150));
    }
    test.expect(protocol_ok, "four-rank weighted/sum combine, backpressure, and sentinels");
    if (protocol_ok)
        wait_children(children, 3, std::chrono::seconds(8), test);
    else
        terminate_children(children, 3);
}

void test_negative_protocol(TestContext& test) {
    Window window{};
    HostSync fast_sync(std::chrono::milliseconds(2));
    std::array<float, kCapacity> source{}, output{};
    fill_payload(source.data(), kCapacity, 1, 0);
    output.fill(kSentinel);

    test.expect(supernpu::comm::send<CpuPayload>(window, source.data(), kWidth, 0, fast_sync) ==
                    Status::invalid_argument,
                "send rejects epoch zero");
    test.expect(supernpu::comm::send<CpuPayload>(window, source.data(), kWidth + 1, 1, fast_sync) ==
                    Status::invalid_argument,
                "send rejects partial tile");
    test.expect(supernpu::comm::receive<CpuPayload>(nullptr, window, kWidth, 1, fast_sync) ==
                    Status::invalid_argument,
                "receive rejects null output");

    const auto before = output;
    test.expect(supernpu::comm::receive<CpuPayload>(output.data(), window, kWidth, 1, fast_sync) ==
                    Status::timeout,
                "receive times out on stale ready");
    test.expect(output == before && window.ready == 0 && window.consumed == 0,
                "receive timeout does not modify output or acknowledge");

    test.expect(supernpu::comm::send<CpuPayload>(window, source.data(), kWidth, 1, fast_sync) == Status::ok,
                "first send succeeds");
    const std::uint64_t ready_before_duplicate = window.ready;
    std::array<float, kCapacity> payload_before_duplicate{};
    std::memcpy(payload_before_duplicate.data(), window.data, sizeof(window.data));
    test.expect(supernpu::comm::send<CpuPayload>(window, source.data(), kWidth, 1, fast_sync) ==
                    Status::timeout,
                "duplicate epoch send times out");
    test.expect(window.ready == ready_before_duplicate && window.consumed == 0 &&
                    std::memcmp(payload_before_duplicate.data(), window.data,
                                sizeof(window.data)) == 0,
                "duplicate send does not alter payload, republish, or acknowledge");

    output.fill(kSentinel);
    test.expect(supernpu::comm::receive<CpuPayload>(output.data(), window, kWidth * 2, 1, fast_sync) ==
                    Status::invalid_argument,
                "receive rejects published count mismatch");
    test.expect(window.consumed == 0, "count mismatch does not acknowledge");
    test.expect(supernpu::comm::receive<CpuPayload>(output.data(), window, kWidth, 1, fast_sync) == Status::ok,
                "matching receive completes the epoch");

    Window* duplicate[] = {&window, &window};
    Window* with_null[] = {&window, nullptr};
    test.expect(supernpu::comm::receive_combine<CpuPayload>(
                    output.data(), duplicate, nullptr, 2, kWidth, 2, fast_sync) ==
                    Status::invalid_argument,
                "combine rejects duplicate windows");
    test.expect(supernpu::comm::receive_combine<CpuPayload>(
                    output.data(), with_null, nullptr, 2, kWidth, 2, fast_sync) ==
                    Status::invalid_argument,
                "combine rejects a null window");
    test.expect(supernpu::comm::receive_combine<CpuPayload>(
                    output.data(), duplicate, nullptr, 0, kWidth, 2, fast_sync) ==
                    Status::invalid_argument,
                "combine rejects zero windows");
}

}  // namespace

int main() {
    TestContext test;
    test_direct_operations(test);
    test_single_producer_protocol(test);
    test_four_rank_combine(test);
    test_negative_protocol(test);
    if (test.failures != 0) {
        std::fprintf(stderr, "host_mmap_test: %d failure(s)\n", test.failures);
        return 1;
    }
    std::puts("PASS host_mmap_test: mapped Put/Get, weighted combine, sum, epochs, and failures");
    return 0;
}
