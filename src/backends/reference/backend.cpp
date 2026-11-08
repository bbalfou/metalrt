// Explicit CPU validation backend. It validates scheduling semantics, not GPU speed.
// Deterministic semantic validation backend. CPU callbacks implement kernel behavior;
// no MSL compilation or native GPU scheduling is simulated here.
#include <metalrt/detail/backend.h>
#include <algorithm>
#include <condition_variable>
#include <deque>
#include <future>
#include <new>
#include <thread>

namespace metalrt {
namespace detail {
namespace {
struct ReferenceBuffer final : BufferAllocation {
    void* memory = nullptr;
    ReferenceBuffer(size_t n, StorageMode s, const void* device) {
        bytes = n;
        storage = s;
        device_identity = device;
        memory = ::operator new(n, std::align_val_t(256));
    }
    ~ReferenceBuffer() override { ::operator delete(memory, std::align_val_t(256)); }
    void* contents() const override { return memory; }
};
struct ReferenceCompletion final : Completion {
    std::shared_future<void> future;
    explicit ReferenceCompletion(std::shared_future<void> f) : future(std::move(f)) {}
    void wait() override { future.get(); }
    bool ready() const override {
        return future.wait_for(std::chrono::seconds(0)) == std::future_status::ready;
    }
};
struct ReferenceEvent final : EventState {
    std::atomic<uint64_t> completed{0};
    uint64_t completed_value() const override { return completed.load(std::memory_order_acquire); }
};
struct ReferenceICB {
    std::vector<Command> commands;
    size_t buffer_slots = 0, threadgroup_slots = 0;
};
struct ReferenceSegment {
    size_t begin = 0, count = 0, buffer_slots = 0, threadgroup_slots = 0;
    std::shared_ptr<ReferenceICB> storage;
};
struct ReferencePlan final : Plan {
    std::mutex mutex;
    std::vector<Command> commands;
    std::vector<ReferenceSegment> segments;
    DispatchPolicy policy = DispatchPolicy::Direct;
    size_t inflight = 0; // Protected by mutex, including queues launching the same graph.
};
bool static_compute(const Command& command) {
    return command.kind == Command::Kind::Compute && !command.compute.indirect_groups &&
           !command.compute.predicate && !command.compute.dispatch_status;
}
std::vector<ReferenceSegment> segment_layout(const std::vector<Command>& commands) {
    std::vector<ReferenceSegment> result;
    for (size_t i = 0; i < commands.size();) {
        if (!static_compute(commands[i])) {
            ++i;
            continue;
        }
        ReferenceSegment segment;
        segment.begin = i;
        while (i < commands.size() && static_compute(commands[i]) && segment.count < 16384) {
            const auto& compute = commands[i].compute;
            if (compute.optional_buffer_mask != UINT32_MAX)
                segment.buffer_slots =
                    std::max(segment.buffer_slots, size_t(compute.optional_buffer_mask) + 1);
            for (const auto& mapping : compute.pointer_base_arguments)
                segment.buffer_slots = std::max(segment.buffer_slots, size_t(mapping.second) + 1);
            for (const auto& argument : commands[i].compute.arguments) {
                auto& slots = argument.kind == Argument::Kind::Threadgroup
                                  ? segment.threadgroup_slots
                                  : segment.buffer_slots;
                slots = std::max(slots, size_t(argument.index) + 1);
            }
            ++i;
            ++segment.count;
        }
        result.push_back(std::move(segment));
    }
    return result;
}
std::shared_ptr<ReferencePlan> prepare_plan(const std::vector<Command>& commands,
                                            DispatchPolicy policy,
                                            const std::shared_ptr<Plan>& cached,
                                            Statistics& statistics,
                                            std::unique_lock<std::mutex>& result_lock) {
    auto previous = std::dynamic_pointer_cast<ReferencePlan>(cached);
    std::unique_lock<std::mutex> previous_lock;
    if (previous) {
        previous_lock = std::unique_lock<std::mutex>(previous->mutex);
        if (previous->policy != policy) {
            previous_lock.unlock();
            previous.reset();
        }
    }
    if (previous && same_commands(previous->commands, commands)) {
        statistics.icb_reuses = previous->segments.size();
        result_lock = std::move(previous_lock);
        return previous;
    }
    auto replacement = commands;
    auto layout = policy == DispatchPolicy::Indirect ? segment_layout(commands)
                                                     : std::vector<ReferenceSegment>{};
    const bool idle = previous && previous->inflight == 0;
    auto plan = idle ? previous : std::make_shared<ReferencePlan>();
    if (idle)
        result_lock = std::move(previous_lock);
    else
        result_lock = std::unique_lock<std::mutex>(plan->mutex);
    for (auto& segment : layout) {
        const ReferenceSegment* old = nullptr;
        if (previous)
            for (const auto& candidate : previous->segments)
                if (candidate.begin == segment.begin && candidate.count == segment.count) {
                    old = &candidate;
                    break;
                }
        std::vector<Command> content{commands.begin() + segment.begin,
                                     commands.begin() + segment.begin + segment.count};
        if (old && same_commands(old->storage->commands, content)) {
            segment.storage = old->storage;
            ++statistics.icb_reuses;
            continue;
        }
        auto storage = std::make_shared<ReferenceICB>();
        storage->commands = std::move(content);
        storage->buffer_slots = segment.buffer_slots;
        storage->threadgroup_slots = segment.threadgroup_slots;
        if (idle && old && old->storage.use_count() == 1 &&
            old->storage->buffer_slots >= segment.buffer_slots &&
            old->storage->threadgroup_slots >= segment.threadgroup_slots) {
            storage->buffer_slots = old->storage->buffer_slots;
            storage->threadgroup_slots = old->storage->threadgroup_slots;
            ++statistics.icb_updates;
        } else
            ++statistics.icb_builds;
        segment.storage = std::move(storage);
    }
    plan->commands = std::move(replacement);
    plan->segments = std::move(layout);
    plan->policy = policy;
    return plan;
}
void* memory(const BufferView& view) {
    return static_cast<uint8_t*>(view.buffer.implementation()->contents()) + view.offset;
}
void execute(const Command& command) {
    if (command.kind == Command::Kind::Compute) {
        auto dispatch = command.compute;
        uint32_t status = 0;
        bool active = true;
        if (dispatch.indirect_groups) {
            uint32_t requested[3];
            std::memcpy(requested, memory(dispatch.indirect_groups), sizeof(requested));
            const size_t maximum[3] = {dispatch.max_groups.x, dispatch.max_groups.y,
                                       dispatch.max_groups.z};
            size_t actual[3];
            for (size_t axis = 0; axis < 3; ++axis) {
                if (requested[axis] > maximum[axis])
                    status |= 1u;
                actual[axis] = std::min(size_t(requested[axis]), maximum[axis]);
                active = active && actual[axis] != 0;
            }
            dispatch.grid = {checked_mul(actual[0], dispatch.group.x),
                             checked_mul(actual[1], dispatch.group.y),
                             checked_mul(actual[2], dispatch.group.z)};
        }
        if (dispatch.predicate) {
            uint32_t enabled;
            std::memcpy(&enabled, memory(dispatch.predicate), sizeof(enabled));
            if (!enabled) {
                active = false;
                status |= 2u;
            }
        }
        if (dispatch.dispatch_status)
            std::memcpy(memory(dispatch.dispatch_status), &status, sizeof(status));
        if (!active)
            return;
        if (dispatch.optional_buffer_mask != UINT32_MAX ||
            !dispatch.pointer_base_arguments.empty()) {
            dispatch.arguments = materialize_compute_arguments(dispatch);
            // The callback sees complete immutable arguments, including the
            // derived scalar, while semantic NullBuffer values stay nullptr.
            dispatch.optional_buffer_mask = UINT32_MAX;
            dispatch.pointer_base_arguments.clear();
        }
        const auto& kernel = *dispatch.kernel.implementation();
        if (!kernel.reference)
            throw std::runtime_error("no CPU reference implementation for kernel " + kernel.entry);
        kernel.reference(dispatch);
    } else if (command.kind == Command::Kind::Copy) {
        if (command.source == command.destination)
            return;
        std::memcpy(memory(command.destination), memory(command.source), command.source.length);
    } else
        std::memset(memory(command.destination), command.fill_value, command.destination.length);
}
void execute_sequence(const std::vector<Command>& commands, bool continue_after_error) {
    std::exception_ptr first_error;
    for (size_t begin = 0; begin < commands.size();) {
        size_t end = begin + 1;
        if (!continue_after_error && static_compute(commands[begin]))
            while (end < commands.size() && static_compute(commands[end]) &&
                   !commands[end].compute.barrier_before)
                ++end;
        try {
            if (end == begin + 1)
                execute(commands[begin]);
            else {
                // A barrier joins the whole preceding group. Launching actual
                // CPU tasks makes missed joins observable in deterministic tests.
                std::vector<std::future<void>> tasks;
                tasks.reserve(end - begin);
                std::exception_ptr group_error;
                try {
                    for (size_t i = begin; i < end; ++i)
                        tasks.push_back(std::async(std::launch::async,
                                                   [&commands, i] { execute(commands[i]); }));
                } catch (...) {
                    group_error = std::current_exception();
                }
                for (auto& task : tasks)
                    try {
                        task.get();
                    } catch (...) {
                        if (!group_error)
                            group_error = std::current_exception();
                    }
                if (group_error)
                    std::rethrow_exception(group_error);
            }
        } catch (...) {
            if (!first_error)
                first_error = std::current_exception();
            if (!continue_after_error)
                break;
        }
        begin = end;
    }
    if (first_error)
        std::rethrow_exception(first_error);
}
struct ReferenceQueue final : QueueBackend {
    // One queue node owns the full submission. PerKernel still represents one
    // logical command buffer per command, but the all-or-nothing queue insertion
    // cannot leave an untracked submitted prefix if an allocation throws.
    struct Work {
        std::vector<Command> commands;
        std::shared_ptr<ReferencePlan> plan;
        SubmitOptions options;
        DispatchPolicy policy = DispatchPolicy::Direct;
        std::promise<void> result;
    };
    std::mutex mutex;
    std::condition_variable available;
    std::deque<std::unique_ptr<Work>> pending;
    bool stopping = false;
    std::thread worker;
    ReferenceQueue() : worker([this] { run(); }) {}
    ~ReferenceQueue() override {
        {
            std::lock_guard<std::mutex> lock(mutex);
            stopping = true;
        }
        available.notify_one();
        worker.join();
    }
    void run() {
        for (;;) {
            std::unique_ptr<Work> work;
            {
                std::unique_lock<std::mutex> lock(mutex);
                available.wait(lock, [this] { return stopping || !pending.empty(); });
                if (pending.empty())
                    return;
                work = std::move(pending.front());
                pending.pop_front();
            }
            std::exception_ptr error;
            try {
                // These waits occur on the queue worker, never on the submitting
                // thread. EventPoints carry producer-prefix errors as well.
                for (const auto& point : work->options.waits)
                    point.wait();
                execute_sequence(work->commands, work->policy == DispatchPolicy::PerKernel);
            } catch (...) {
                error = std::current_exception();
            }
            if (work->options.signal_event) {
                auto event = std::static_pointer_cast<ReferenceEvent>(work->options.signal_event);
                event->completed.store(work->options.signal_value, std::memory_order_release);
            }
            for (const auto& signal : work->options.signals) {
                auto event = std::static_pointer_cast<ReferenceEvent>(signal.event);
                event->completed.store(signal.value, std::memory_order_release);
            }
            // Commands and their retained allocations are gone before completion
            // is published. An idle plan may then be updated without racing work.
            work->commands.clear();
            work->options = {};
            if (work->plan) {
                {
                    std::lock_guard<std::mutex> lock(work->plan->mutex);
                    --work->plan->inflight;
                }
                work->plan.reset();
            }
            if (error)
                work->result.set_exception(error);
            else
                work->result.set_value();
        }
    }
    SubmitResult submit(const std::vector<Command>& commands, DispatchPolicy policy,
                        std::shared_ptr<Plan> cached, const SubmitOptions& options) override {
        SubmitResult result;
        if (commands.empty() && options.waits.empty() && !options.signal_event &&
            options.signals.empty() && !options.prepare_only)
            return result;
        std::unique_lock<std::mutex> plan_lock;
        auto plan = prepare_plan(commands, policy, cached, result.statistics, plan_lock);
        result.plan = plan;
        if (options.prepare_only) {
            result.statistics.prepare_calls = 1;
            return result;
        }
        for (const auto& command : commands) {
            if (command.kind == Command::Kind::Compute) {
                if (command.compute.control)
                    ++result.statistics.control_dispatches;
                else
                    ++result.statistics.compute_dispatches;
                if (!static_compute(command))
                    ++result.statistics.control_dispatches;
            } else
                ++result.statistics.blit_commands;
        }
        result.statistics.command_buffers =
            policy == DispatchPolicy::PerKernel ? std::max(size_t(1), commands.size()) : 1;
        result.statistics.event_waits = options.waits.size();
        result.statistics.event_signals = (options.signal_event ? 1 : 0) + options.signals.size();
        auto work = std::make_unique<Work>();
        work->commands = plan->commands;
        work->plan = plan;
        work->options = options;
        work->policy = policy;
        result.completion = make_completion<ReferenceCompletion>(work->result.get_future().share());
        {
            std::lock_guard<std::mutex> lock(mutex);
            pending.push_back(std::move(work));
            ++plan->inflight;
        }
        plan_lock.unlock();
        available.notify_one();
        return result;
    }
};
struct ReferenceDevice final : DeviceBackend {
    Capabilities capabilities() const override {
        return {BackendKind::Reference, "CPU reference (validation only)", true, true};
    }
    std::shared_ptr<BufferAllocation> allocate(size_t n, StorageMode s) override {
        return std::make_shared<ReferenceBuffer>(n, s, this);
    }
    std::shared_ptr<KernelState> compile(const std::string&, const std::string& entry,
                                         CpuKernel cpu) override {
        if (!cpu)
            throw std::invalid_argument("CPU reference backend requires a reference function for " +
                                        entry);
        auto kernel = std::make_shared<KernelState>();
        kernel->entry = entry;
        kernel->reference = std::move(cpu);
        kernel->device_identity = this;
        ++kernel_compilations;
        return kernel;
    }
    std::unique_ptr<QueueBackend> make_queue() override {
        return std::make_unique<ReferenceQueue>();
    }
    std::shared_ptr<EventState> make_event() override { return std::make_shared<ReferenceEvent>(); }
};
} // namespace
std::shared_ptr<DeviceBackend> make_reference_device() {
    return std::make_shared<ReferenceDevice>();
}
} // namespace detail
} // namespace metalrt
