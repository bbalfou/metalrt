#pragma once
// Shared runtime state and recording invariants. This header is private to the
// library: adapters should compose Stream/Graph rather than depend on these types.
#include <metalrt/detail/backend.h>
#include <algorithm>
#include <metalrt/interop.h>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <thread>
#include <unordered_set>
#include <utility>

// Lock order: device ordering domain, then stream, then graph. Recording state
// is owned by its initiating host thread; cancellation must invalidate descendants.
namespace metalrt {
inline void require(bool condition, const char* message) {
    if (!condition)
        throw std::invalid_argument(message);
}
extern std::mutex streams_mutex;
extern std::vector<std::weak_ptr<detail::StreamState>> streams;
namespace detail {
extern std::atomic<uint64_t> next_stream_id;
struct PrefixCompletion final : Completion {
    std::vector<Ticket> tickets;
    explicit PrefixCompletion(std::vector<Ticket> prefix) : tickets(std::move(prefix)) {}
    bool append_dependencies(std::vector<Ticket>& dependencies, bool) override {
        dependencies.insert(dependencies.end(), tickets.begin(), tickets.end());
        return true;
    }
    void wait() override {}
    bool ready() const override { return true; }
};
struct OrderingDomain {
    std::recursive_mutex mutex;
    std::vector<std::weak_ptr<StreamState>> streams;
    std::vector<std::shared_ptr<OrderedSubmission>> pending;
    std::weak_ptr<OrderedSubmission> last_issued;
};
// A submitted operation may depend on a still-cancellable recording in another
// queue. This completion becomes attached to actual backend work at commit; it
// never claims that a reservation has executed successfully.
struct DeferredCompletion final : Completion {
    mutable std::mutex mutex;
    std::condition_variable available;
    bool resolved = false;
    std::shared_ptr<Completion> actual;
    std::exception_ptr failure;
    Ticket prefix;
    std::function<bool()> recording_dependency;
    void resolve(std::shared_ptr<Completion> value = {}, std::exception_ptr error = {}) {
        {
            std::lock_guard<std::mutex> lock(mutex);
            actual = std::move(value);
            failure = error;
            resolved = true;
        }
        available.notify_all();
    }
    bool append_dependencies(std::vector<Ticket>& dependencies, bool waiting) override {
        if (waiting && recording_dependency && recording_dependency())
            throw RecordingDependencyError(
                "cannot wait on work that depends on this thread's unfinished recording; finish or "
                "cancel that recording first");
        std::unique_lock<std::mutex> lock(mutex);
        if (waiting)
            available.wait(lock, [this] { return resolved; });
        else if (!resolved)
            return false;
        if (prefix)
            dependencies.push_back(prefix);
        if (actual)
            dependencies.emplace_back(actual);
        return true;
    }
    void wait() override {
        std::lock_guard<std::mutex> lock(mutex);
        if (failure)
            std::rethrow_exception(failure);
    }
    bool ready() const override {
        std::lock_guard<std::mutex> lock(mutex);
        return resolved && (!actual || actual->ready());
    }
};
struct OrderedSubmission {
    std::shared_ptr<StreamState> owner; // Released after submission, not before.
    std::vector<Command> commands;
    SubmitOptions options;
    DispatchPolicy policy = DispatchPolicy::Direct;
    std::shared_ptr<Plan> plan;
    std::shared_ptr<DeferredCompletion> completion;
    EventPoint point;
    std::vector<std::shared_ptr<OrderedSubmission>> dependencies;
    size_t pending_begin = 0;
    std::thread::id recording_thread;
    bool committed = false, cancelled = false, submitted = false, observed = false;
    std::exception_ptr failure;
};
struct StreamState : std::enable_shared_from_this<StreamState> {
    std::atomic<uint64_t> client_handles{1};
    Device device;
    DispatchPolicy policy;
    std::unique_ptr<QueueBackend> queue;
    mutable std::recursive_mutex mutex;
    std::shared_ptr<OrderingDomain> domain;
    StreamOrdering ordering = StreamOrdering::Independent;
    std::shared_ptr<EventState> ordering_event;
    uint64_t ordering_value = 0;
    std::shared_ptr<OrderedSubmission> ordering_tail;
    std::unordered_map<uint64_t, uint64_t> ordering_seen;
    std::weak_ptr<OrderedSubmission> last_submission;
    std::vector<std::shared_ptr<OrderedSubmission>> pending_ordered;
    enum class Recording { None, Batch, Capture } recording = Recording::None;
    std::thread::id recording_thread;
    uint64_t generation = 0;
    uint64_t id = next_stream_id++, parallel_id = 0, next_parallel_id = 0;
    uint64_t next_scope_id = 0;
    std::vector<uint64_t> nested_scopes;
    bool parallel_active = false, parallel_first = false, force_join = false;
    BufferView predicate;
    std::vector<Command> pending;
    std::vector<Ticket> outstanding;
    Ticket last;
    Statistics stats;
    std::shared_ptr<TimingState> timing;
    bool timing_collecting = false;
    std::thread::id timing_thread;
    uint64_t active_synchronizers = 0;
    std::exception_ptr error;
    void check_thread() const {
        if (recording != Recording::None && recording_thread != std::this_thread::get_id())
            throw std::logic_error("a recording stream must be used by its capturing thread");
        if (timing && timing_thread != std::this_thread::get_id())
            throw std::logic_error("a profiled stream must be used by its profiling thread");
    }
    void check_error() {
        if (error) {
            auto e = std::exchange(error, {});
            std::rethrow_exception(e);
        }
    }
    SubmitResult backend_submit(const std::vector<Command>& commands, const SubmitOptions& options,
                                DispatchPolicy selected, std::shared_ptr<Plan> cached = {}) {
        for (const auto& point : options.waits) {
            require(static_cast<bool>(point), "cannot wait on an empty event point");
            require(point.implementation()->device_identity == device.implementation().get(),
                    "event and waiting stream belong to different devices");
        }
        // Queue completion is ordered. Once the first retained prefix is still
        // pending, probing every later prefix only repeats the same graph walk.
        auto retired = outstanding.begin();
        while (retired != outstanding.end() && retired->ready()) {
            retired->wait();
            ++retired;
        }
        outstanding.erase(outstanding.begin(), retired);
        outstanding.reserve(checked_add(outstanding.size(), 1));
        const auto begin = std::chrono::steady_clock::now();
        auto result = queue->submit(commands, selected, std::move(cached), options);
        stats.cpu_submit_nanoseconds +=
            static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                      std::chrono::steady_clock::now() - begin)
                                      .count());
        stats += result.statistics;
        return result;
    }
    bool ordered() const { return ordering != StreamOrdering::Independent; }
    std::shared_ptr<OrderedSubmission> reserve_ordered(std::vector<Command>, const SubmitOptions&,
                                                       bool committed,
                                                       std::shared_ptr<Plan> plan = {});
    void submit(const std::vector<Command>& commands, std::shared_ptr<Plan>* cache = nullptr,
                const SubmitOptions& options = {}, bool boundary = false);
    void record_ordered(const Command& command);
    void commit_ordered();
    void cancel_ordered(size_t keep);
    void publish_ordered() {
        // The ordered tail retains the entire preceding stream prefix. Keeping
        // every prefix again would make a synchronization repeatedly traverse
        // the same completion DAG and retain redundant ownership.
        outstanding.clear();
        last = ordering_tail->point.completion();
        outstanding.push_back(last);
    }
};
struct GraphState {
    mutable std::mutex mutex;
    std::vector<Command> commands;
    std::shared_ptr<Plan> plan;
    Device device;
    bool capturing = false, synchronize = false;
    uint64_t capture_owner = 0, capture_generation = 0;
    uint64_t revision = 0;
};

std::shared_ptr<OrderingDomain> ordering_domain(const Device& device);
bool blocked_by_recording(const std::shared_ptr<OrderedSubmission>& node, std::thread::id thread);
void drain_ordered(const std::shared_ptr<OrderingDomain>& domain);
} // namespace detail
// A cancelled scope removes its entire recorded suffix and invalidates all of
// its descendants. Scope IDs are never reused within a recording generation.
struct RecordingCheckpoint {
    size_t pending = 0;
    uint64_t generation = 0, scope = 0, parallel_id = 0, next_parallel_id = 0;
    bool parallel_active = false, parallel_first = false, force_join = false;
    BufferView predicate;
};

RecordingCheckpoint checkpoint(const detail::StreamState& s);
RecordingCheckpoint begin_nested_scope(detail::StreamState& s);
bool finish_nested_scope(detail::StreamState& s, const RecordingCheckpoint& saved);
void restore_checkpoint(detail::StreamState& s, const RecordingCheckpoint& saved);
void cancel_nested_scope(detail::StreamState& s, const RecordingCheckpoint& saved);
void end_recording(detail::StreamState& s);
void validate_buffer(const BufferView& view, const Device& device);
void validate_command(const Command& c, const Device& d);
void validate_sequence(const std::vector<Command>& commands, const Device& device);
void validate_parallel_pair(const ComputeCommand& a, const ComputeCommand& b);
void append_command(detail::StreamState& s, Command command);
void defer_error(const std::shared_ptr<detail::StreamState>& s, std::exception_ptr error);
void wait_stream(const std::shared_ptr<detail::StreamState>& s, bool allow_recording = false);
} // namespace metalrt
