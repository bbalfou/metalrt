#include <metalrt/runtime.h>
#include <metalrt/detail/backend.h>
#include <chrono>
#include <cmath>
#include <deque>
#include <future>
#include <iostream>
#include <limits>
#include <string>

namespace {
void check(bool condition, const char* text) {
    if (!condition)
        throw std::runtime_error(text);
}

template <class F> void rejects(F fn, const char* text) {
    bool rejected = false;
    try {
        fn();
    } catch (const std::exception&) {
        rejected = true;
    }
    check(rejected, text);
}

// A deterministic test backend exposes the interval after synchronize() has
// taken ownership of outstanding tickets but before those tickets finish.
// It reports Reference capabilities and never invents hardware timestamps.
struct ProbeCompletion final : metalrt::detail::Completion {
    std::promise<void> entered_promise, release_promise;
    std::shared_future<void> entered = entered_promise.get_future().share();
    std::shared_future<void> released = release_promise.get_future().share();
    std::once_flag announced, opened;
    std::shared_ptr<metalrt::detail::TimingState> submitted_timing;
    bool fail = false;

    explicit ProbeCompletion(bool blocked = false, bool failure = false) : fail(failure) {
        if (!blocked)
            release();
    }

    void release() {
        std::call_once(opened, [&] { release_promise.set_value(); });
    }

    void wait() override {
        std::call_once(announced, [&] { entered_promise.set_value(); });
        released.wait();
        if (fail)
            throw std::runtime_error("controlled completion failure");
    }

    bool ready() const override {
        return released.wait_for(std::chrono::seconds(0)) == std::future_status::ready;
    }
};

using ProbeWork = std::deque<std::shared_ptr<ProbeCompletion>>;

struct ProbeQueue final : metalrt::detail::QueueBackend {
    std::shared_ptr<ProbeWork> pending;

    explicit ProbeQueue(std::shared_ptr<ProbeWork> work) : pending(std::move(work)) {}

    metalrt::detail::SubmitResult submit(const std::vector<metalrt::Command>&,
                                         metalrt::DispatchPolicy,
                                         std::shared_ptr<metalrt::detail::Plan>,
                                         const metalrt::detail::SubmitOptions& options) override {
        check(!pending->empty(), "test backend has no prepared completion");
        auto completion = pending->front();
        pending->pop_front();
        completion->submitted_timing = options.timing;
        metalrt::detail::SubmitResult result;
        result.completion = std::move(completion);
        result.statistics.command_buffers = 1;
        return result;
    }
};

struct ProbeBackend final : metalrt::detail::DeviceBackend {
    std::shared_ptr<ProbeWork> pending = std::make_shared<ProbeWork>();

    metalrt::Capabilities capabilities() const override {
        return {metalrt::BackendKind::Reference, "controlled completion test backend", false, true};
    }

    std::shared_ptr<metalrt::detail::BufferAllocation> allocate(size_t,
                                                                metalrt::StorageMode) override {
        throw std::logic_error("test backend does not allocate buffers");
    }

    std::shared_ptr<metalrt::detail::KernelState>
    compile(const std::string&, const std::string& entry, metalrt::CpuKernel) override {
        auto kernel = std::make_shared<metalrt::detail::KernelState>();
        kernel->entry = entry;
        kernel->device_identity = this;
        return kernel;
    }

    std::unique_ptr<metalrt::detail::QueueBackend> make_queue() override {
        return std::make_unique<ProbeQueue>(pending);
    }

    std::shared_ptr<metalrt::detail::EventState> make_event() override {
        struct Event final : metalrt::detail::EventState {
            uint64_t completed_value() const override { return 0; }
        };
        return std::make_shared<Event>();
    }
};

struct ProbeStream {
    std::shared_ptr<ProbeBackend> backend = std::make_shared<ProbeBackend>();
    metalrt::Device device{backend};
    metalrt::Stream stream{device};
    metalrt::Kernel kernel = device.kernel("", "controlled_completion");

    void submit(const std::shared_ptr<ProbeCompletion>& completion) {
        backend->pending->push_back(completion);
        stream.dispatch(kernel, {1, 1, 1}, {1, 1, 1}, {});
    }
};

void timing_start_handoff() {
    {
        // The initial drain can issue a native ordering marker even when the
        // profiling queue itself was idle. That pre-window marker is unmeasured.
        auto backend = std::make_shared<ProbeBackend>();
        metalrt::Device device{backend};
        metalrt::Stream producer{device, metalrt::DispatchPolicy::Direct,
                                 metalrt::StreamOrdering::Blocking};
        metalrt::Stream measured{device, metalrt::DispatchPolicy::Direct,
                                 metalrt::StreamOrdering::Default};
        auto kernel = device.kernel("", "controlled_completion");
        auto old = std::make_shared<ProbeCompletion>();
        auto marker = std::make_shared<ProbeCompletion>();
        auto work = std::make_shared<ProbeCompletion>();
        backend->pending->push_back(old);
        producer.dispatch(kernel, {1, 1, 1}, {1, 1, 1}, {});
        backend->pending->push_back(marker);
        measured.begin_gpu_timing();
        backend->pending->push_back(work);
        measured.dispatch(kernel, {1, 1, 1}, {1, 1, 1}, {});
        measured.end_gpu_timing();
        check(!old->submitted_timing && !marker->submitted_timing && work->submitted_timing,
              "pre-window default-stream drain entered the profiling collector");
        producer.synchronize();
    }
    {
        ProbeStream probe;
        auto previous = std::make_shared<ProbeCompletion>(true);
        probe.submit(previous);
        auto waiter = std::async(std::launch::async, [&] { probe.stream.synchronize(); });
        const bool entered =
            previous->entered.wait_for(std::chrono::seconds(5)) == std::future_status::ready;
        bool rejected = false;
        if (entered) {
            try {
                probe.stream.begin_gpu_timing();
            } catch (const std::exception&) {
                rejected = true;
            }
            if (!rejected)
                probe.stream.end_gpu_timing();
        }
        // Release before assertions, so even the old broken implementation
        // fails with a diagnostic instead of blocking in future destruction.
        previous->release();
        waiter.get();
        check(entered, "controlled synchronizer did not enter its completion wait");
        check(rejected, "profiling began while another synchronizer still owned unfinished work");
        probe.stream.begin_gpu_timing();
        check(!probe.stream.end_gpu_timing().valid(), "rejected profiling start retained a window");
    }
    {
        ProbeStream probe;
        auto warmup = std::make_shared<ProbeCompletion>(true);
        auto measured = std::make_shared<ProbeCompletion>();
        probe.submit(warmup);
        auto owner = std::async(std::launch::async, [&] {
            probe.stream.begin_gpu_timing();
            probe.submit(measured);
            return probe.stream.end_gpu_timing();
        });
        const bool entered =
            warmup->entered.wait_for(std::chrono::seconds(5)) == std::future_status::ready;
        bool policy_rejected = false;
        if (entered) {
            try {
                probe.stream.set_policy(metalrt::DispatchPolicy::PerKernel);
            } catch (const std::exception&) {
                policy_rejected = true;
            }
        }
        warmup->release();
        const auto timing = owner.get();
        check(entered && policy_rejected,
              "profiling did not reserve its owner before draining warmup");
        check(!warmup->submitted_timing && measured->submitted_timing,
              "warmup and measured submissions used the same timing scope");
        check(!timing.supported && !timing.valid() && !timing.command_buffers,
              "controlled Reference completion fabricated GPU time");
    }
    {
        ProbeStream probe;
        probe.submit(std::make_shared<ProbeCompletion>(false, true));
        rejects([&] { probe.stream.begin_gpu_timing(); },
                "warmup completion failure was swallowed");
        // Starting again proves a failed initial drain released its reservation.
        probe.stream.begin_gpu_timing();
        probe.stream.end_gpu_timing();
    }
}

void timestamp_accounting() {
    metalrt::detail::TimingState state;
    state.result.supported = true;
    state.record(10, 12, true);
    state.record(11, 14, true);
    auto result = state.snapshot();
    check(result.valid() && result.command_buffers == 2, "valid timestamp count");
    check(result.command_buffer_seconds == 5 && result.span_seconds() == 4,
          "sum and span must preserve overlap");
    state.record(0, 0, true);
    state.record(20, 19, true);
    state.record(20, std::numeric_limits<double>::infinity(), true);
    state.record(20, 21, false);
    result = state.snapshot();
    check(!result.valid() && result.command_buffers == 6 && result.timed_command_buffers == 2,
          "missing/error timestamps must invalidate completeness");
    check(result.command_buffer_seconds == 5,
          "invalid timestamps must not enter measured duration");
}

void stream_windows(metalrt::Device device, metalrt::DispatchPolicy policy) {
    metalrt::Stream stream{device, policy};
    const auto buffer = device.allocate(65536);
    stream.fill(buffer.view(), 1);
    stream.synchronize(); // Unmeasured warmup.
    const auto before = stream.statistics();
    stream.begin_gpu_timing();
    rejects([&] { stream.begin_gpu_timing(); }, "nested timing windows accepted");
    rejects([&] { stream.set_policy(policy); }, "profiling owner changed its dispatch policy");
    rejects([&] { stream.reset_statistics(); }, "profiling owner reset its statistics");
    auto other = std::async(std::launch::async, [&] {
        rejects([&] { stream.fill(buffer.view(), 99); },
                "profiling accepted another submitting thread");
        rejects([&] { stream.set_policy(policy); },
                "profiling accepted another thread changing policy");
        rejects([&] { stream.reset_statistics(); },
                "profiling accepted another thread resetting statistics");
    });
    other.get();
    for (unsigned i = 0; i < 3; ++i) {
        auto batch = stream.batch();
        stream.fill(buffer.view(), uint8_t(i + 2));
        stream.fill(buffer.view(), uint8_t(i + 3));
        batch.finish();
        if (i == 1) {
            stream.last_ticket().wait();
            stream.last_ticket().wait();
        }
    }
    const auto timing = stream.end_gpu_timing();
    const auto command_buffers = stream.statistics().command_buffers - before.command_buffers;
    check(buffer.data<uint8_t>()[0] == 5, "timed work did not complete");
    if (device.capabilities().backend == metalrt::BackendKind::Metal) {
        check(timing.valid(), "Metal completion timestamps are missing or invalid");
        check(timing.command_buffers == command_buffers,
              "command buffers counted twice or lost during retirement");
        check(timing.command_buffer_seconds > 0 && timing.span_seconds() > 0,
              "Metal GPU interval is not positive");
    } else {
        check(!timing.supported && !timing.valid() && !timing.command_buffers &&
                  timing.command_buffer_seconds == 0,
              "Reference execution fabricated GPU timestamps");
    }
    rejects([&] { stream.end_gpu_timing(); }, "end without begin accepted");
    {
        auto batch = stream.batch();
        rejects([&] { stream.begin_gpu_timing(); }, "begin inside recording accepted");
        batch.cancel();
    }
    stream.begin_gpu_timing();
    {
        auto batch = stream.batch();
        rejects([&] { stream.end_gpu_timing(); }, "end inside recording accepted");
        batch.cancel();
    }
    const auto empty = stream.end_gpu_timing();
    check(!empty.valid() && !empty.command_buffers, "empty window inherited earlier timestamps");
    metalrt::Graph graph;
    {
        auto record = graph.record_guard(&stream);
        stream.fill(buffer.view(), 7);
        record.finish();
    }
    graph.prepare(&stream);
    stream.begin_gpu_timing();
    graph.launch(&stream).wait();
    graph.launch(&stream).wait();
    const auto replay = stream.end_gpu_timing();
    check(buffer.data<uint8_t>()[0] == 7, "profiled graph replay did not execute");
    if (device.capabilities().backend == metalrt::BackendKind::Metal)
        check(replay.valid() && replay.command_buffers == 2, "prepared graph timing incorrect");
}
} // namespace

int main(int argc, char** argv) {
    try {
        bool native = false;
        if (argc == 2 && std::string(argv[1]) == "--metal")
            native = true;
        else if (argc != 1)
            throw std::invalid_argument("usage: test_timing [--metal]");
        timestamp_accounting();
        timing_start_handoff();
        metalrt::Device device;
        if (native) {
            try {
                device = metalrt::Device::system_default();
            } catch (const std::exception& e) {
                std::cout << "SKIP: " << e.what() << '\n';
                return 77;
            }
        } else
            device = metalrt::Device::reference();
        for (auto policy : {metalrt::DispatchPolicy::PerKernel, metalrt::DispatchPolicy::Direct,
                            metalrt::DispatchPolicy::Indirect}) {
            if (policy == metalrt::DispatchPolicy::Indirect &&
                !device.capabilities().indirect_compute)
                continue;
            stream_windows(device, policy);
        }
        std::cout << "PASS: completion timestamps, window isolation, overlap semantics, repeated "
                     "waits, graph replay and unavailable metrics\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
