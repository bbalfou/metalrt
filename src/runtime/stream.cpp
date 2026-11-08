// Queue ownership, synchronization, events, and dispatch entry points.
#include "runtime_internal.h"

namespace metalrt {
std::mutex streams_mutex;
std::vector<std::weak_ptr<detail::StreamState>> streams;
void wait_stream(const std::shared_ptr<detail::StreamState>& s, bool allow_recording) {
    const auto begin = std::chrono::steady_clock::now();
    std::vector<Ticket> work;
    std::exception_ptr error;
    bool retry_after_commit = false;
    {
        std::lock_guard<std::recursive_mutex> order(s->domain->mutex);
        std::lock_guard<std::recursive_mutex> lock(s->mutex);
        s->check_thread();
        if (!allow_recording && s->recording != detail::StreamState::Recording::None)
            throw std::logic_error(
                "cannot synchronize during recording; use a host() boundary in a batch");
        if (s->recording == detail::StreamState::Recording::None) {
            try {
                s->submit({}, nullptr, {}, true);
            } catch (...) {
                error = std::current_exception();
            }
        }
        if (s->ordering_tail &&
            detail::blocked_by_recording(s->ordering_tail, std::this_thread::get_id()))
            throw RecordingDependencyError("cannot synchronize work that depends on this thread's "
                                           "unfinished recording; finish or cancel it first");
        ++s->active_synchronizers;
        auto deferred = std::exchange(s->error, {});
        if (!error)
            error = deferred;
        work.swap(s->outstanding);
    }
    for (const auto& t : work)
        try {
            t.wait();
        } catch (const UncommittedCommandBufferError&) {
            if (!error)
                error = std::current_exception();
            retry_after_commit = true;
            break;
        } catch (...) {
            if (!error)
                error = std::current_exception();
        }
    {
        std::lock_guard<std::recursive_mutex> order(s->domain->mutex);
        std::lock_guard<std::recursive_mutex> lock(s->mutex);
        --s->active_synchronizers;
        ++s->stats.stream_wait_calls;
        if (retry_after_commit)
            s->outstanding.insert(s->outstanding.begin(), work.begin(), work.end());
        if (s->ordering_tail && s->ordering_tail->point.completion().ready())
            s->ordering_tail.reset();
        s->stats.cpu_wait_nanoseconds +=
            static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                      std::chrono::steady_clock::now() - begin)
                                      .count());
    }
    if (error)
        std::rethrow_exception(error);
}
std::unique_ptr<detail::QueueBackend> make_owned_queue(const Device& device,
                                                       DispatchPolicy policy) {
    // Preserve ordinary Stream construction's validation-before-allocation
    // behavior while sharing state setup with the new borrowed-queue path.
    require(static_cast<bool>(device), "empty stream device");
    if (policy == DispatchPolicy::Indirect && !device.capabilities().indirect_compute)
        throw std::runtime_error(
            "device does not support indirect compute; choose Direct explicitly");
    return device.implementation()->make_queue();
}
Stream::Stream(Device d, DispatchPolicy p) : Stream(std::move(d), p, StreamOrdering::Independent) {}
Stream::Stream(Device d, DispatchPolicy p, StreamOrdering ordering)
    : Stream(d, p, ordering, make_owned_queue(d, p)) {}
Stream::Stream(Device d, DispatchPolicy p, StreamOrdering ordering,
               std::unique_ptr<detail::QueueBackend> queue)
    : state_(std::make_shared<detail::StreamState>()) {
    require(static_cast<bool>(d), "empty stream device");
    require(static_cast<bool>(queue), "empty stream queue");
    if (p == DispatchPolicy::Indirect && !d.capabilities().indirect_compute)
        throw std::runtime_error(
            "device does not support indirect compute; choose Direct explicitly");
    state_->device = std::move(d);
    state_->policy = p;
    state_->queue = std::move(queue);
    state_->domain = detail::ordering_domain(state_->device);
    state_->ordering = ordering;
    {
        std::lock_guard<std::recursive_mutex> order(state_->domain->mutex);
        state_->domain->streams.push_back(state_);
    }
    std::lock_guard<std::mutex> lock(streams_mutex);
    streams.push_back(state_);
}
namespace detail {
struct MetalInterop {
    static std::shared_ptr<Stream> borrow(Device device, std::function<void*()> provider,
                                          DispatchPolicy policy) {
        auto queue = device.implementation()->make_external_metal_queue(std::move(provider));
        return std::shared_ptr<Stream>{
            new Stream(std::move(device), policy, StreamOrdering::Independent, std::move(queue))};
    }
};
} // namespace detail
std::shared_ptr<Stream> borrow_metal_stream(Device device, std::function<void*()> provider,
                                            DispatchPolicy policy) {
    require(static_cast<bool>(device), "empty stream device");
    require(static_cast<bool>(provider),
            "borrowed Metal streams require a command-buffer provider");
    return detail::MetalInterop::borrow(std::move(device), std::move(provider), policy);
}
Stream::Stream(std::shared_ptr<detail::StreamState> state, bool retained_alias)
    : state_(std::move(state)), retained_alias_(retained_alias) {
    state_->client_handles.fetch_add(1, std::memory_order_relaxed);
}
std::shared_ptr<Stream> Stream::retain() const {
    return std::shared_ptr<Stream>{new Stream(state_, true)};
}
Stream::~Stream() noexcept {
    const bool last = state_->client_handles.fetch_sub(1, std::memory_order_acq_rel) == 1;
    if (retained_alias_ && !last)
        return;
    // Its execution owner may be about to commit the current command buffer on
    // this same thread. Native completion handlers own every submitted plan and
    // resource; releasing the facade is deliberately not a synchronization.
    if (state_->queue->externally_managed())
        return;
    try {
        auto s = state_;
        std::vector<Ticket> deferred;
        {
            std::lock_guard<std::recursive_mutex> order(s->domain->mutex);
            std::lock_guard<std::recursive_mutex> lock(s->mutex);
            if (s->ordering_tail &&
                detail::blocked_by_recording(s->ordering_tail, std::this_thread::get_id())) {
                deferred = s->outstanding;
                for (const auto& node : s->pending_ordered)
                    deferred.push_back(node->point.completion());
            }
        }
        if (!deferred.empty()) {
            // Destruction must not deadlock a still-cancellable producer on
            // this thread. Retain the queue and resources until its already-
            // issued work completes. This waiter performs no GPU submission.
            std::thread([s, work = std::move(deferred)] {
                std::exception_ptr error;
                for (const auto& ticket : work)
                    try {
                        ticket.wait();
                    } catch (...) {
                        if (!error)
                            error = std::current_exception();
                    }
                {
                    std::lock_guard<std::recursive_mutex> order(s->domain->mutex);
                    std::lock_guard<std::recursive_mutex> lock(s->mutex);
                    // A retained client may enqueue new work after the original
                    // handle dies. Its pending tail/errors belong to that live
                    // client, not to this snapshot's retirement callback.
                    if (s->client_handles.load(std::memory_order_acquire) == 0) {
                        s->outstanding.clear();
                        s->ordering_tail.reset();
                        if (!error)
                            error = s->error;
                        s->error = {};
                    }
                }
                if (error)
                    try {
                        std::rethrow_exception(error);
                    } catch (const std::exception& e) {
                        std::cerr
                            << "metalrt: unobserved deferred stream error during destruction: "
                            << e.what() << '\n';
                    } catch (...) {
                        std::cerr << "metalrt: unobserved nonstandard deferred stream error during "
                                     "destruction\n";
                    }
            }).detach();
            return;
        }
    } catch (const std::exception& e) {
        std::cerr << "metalrt: could not defer stream lifetime cleanup: " << e.what() << '\n';
    } catch (...) {
        std::cerr << "metalrt: could not defer stream lifetime cleanup\n";
    }
    try {
        wait_stream(state_, true);
    } catch (const std::exception& e) {
        std::cerr << "metalrt: unobserved stream error during destruction: " << e.what() << '\n';
    } catch (...) {
        std::cerr << "metalrt: unobserved nonstandard stream error during destruction\n";
    }
}
Device Stream::device() const {
    return state_->device;
}
StreamOrdering Stream::ordering() const {
    std::lock_guard<std::recursive_mutex> order(state_->domain->mutex);
    return state_->ordering;
}
void Stream::set_ordering(StreamOrdering value) {
    std::lock_guard<std::recursive_mutex> order(state_->domain->mutex);
    std::lock_guard<std::recursive_mutex> lock(state_->mutex);
    state_->check_thread();
    state_->check_error();
    if (state_->ordering == value)
        return;
    state_->ordering = value;
    state_->ordering_seen.clear();
    if (value != StreamOrdering::Independent &&
        state_->recording == detail::StreamState::Recording::Batch && !state_->pending.empty() &&
        state_->pending_ordered.empty()) {
        // A borrowed queue may join while its caller is recording. Its existing
        // independent prefix joins at this registration boundary, stays
        // cancellable, and is never silently flushed by the adapter.
        state_->pending_ordered.reserve(1);
        auto node = state_->reserve_ordered(state_->pending, {}, false);
        node->pending_begin = 0;
        state_->pending_ordered.push_back(std::move(node));
        return;
    }
    // Publish already-issued work when an application-owned queue joins the
    // ordering domain. The marker is queued; registration never waits on the CPU.
    if (value != StreamOrdering::Independent && state_->last) {
        auto node = state_->reserve_ordered({}, {}, true);
        state_->publish_ordered();
        detail::drain_ordered(state_->domain);
        if (node->failure)
            std::rethrow_exception(node->failure);
    }
}
DispatchPolicy Stream::policy() const {
    std::lock_guard<std::recursive_mutex> order(state_->domain->mutex);
    std::lock_guard<std::recursive_mutex> l(state_->mutex);
    return state_->policy;
}
void Stream::set_policy(DispatchPolicy p) {
    std::lock_guard<std::recursive_mutex> order(state_->domain->mutex);
    std::lock_guard<std::recursive_mutex> l(state_->mutex);
    state_->check_thread();
    state_->check_error();
    require(state_->recording == detail::StreamState::Recording::None,
            "cannot change policy while recording");
    require(!state_->timing, "cannot change policy inside a timing window");
    if (p == DispatchPolicy::Indirect && !device().capabilities().indirect_compute)
        throw std::runtime_error("indirect compute unsupported");
    state_->policy = p;
}
void Stream::dispatch(Kernel k, Size3 grid, Size3 group, std::vector<Argument> args) {
    if (!grid.x || !grid.y || !grid.z)
        return;
    Command c;
    c.compute.kernel = std::move(k);
    c.compute.grid = grid;
    c.compute.group = group;
    c.compute.arguments = std::move(args);
    enqueue(std::move(c));
}
void Stream::dispatch_indirect(Kernel kernel, BufferView counts, Size3 maximum, Size3 group,
                               std::vector<Argument> args, BufferView status) {
    Command c;
    c.compute.kernel = std::move(kernel);
    c.compute.group = group;
    c.compute.arguments = std::move(args);
    c.compute.indirect_groups = std::move(counts);
    c.compute.max_groups = maximum;
    c.compute.dispatch_status = std::move(status);
    c.compute.grid = {checked_mul(maximum.x, group.x), checked_mul(maximum.y, group.y),
                      checked_mul(maximum.z, group.z)};
    require(static_cast<bool>(c.compute.indirect_groups),
            "indirect dispatch requires a count buffer");
    enqueue(std::move(c));
}
void Stream::enqueue(Command c) {
    std::lock_guard<std::recursive_mutex> order(state_->domain->mutex);
    std::lock_guard<std::recursive_mutex> l(state_->mutex);
    state_->check_error();
    state_->check_thread();
    append_command(*state_, std::move(c));
}
void Stream::copy(BufferView src, BufferView dst) {
    if (src.length == 0 && dst.length == 0)
        return;
    if (src == dst)
        return;
    Command c;
    c.kind = Command::Kind::Copy;
    c.source = std::move(src);
    c.destination = std::move(dst);
    enqueue(std::move(c));
}
void Stream::fill(BufferView dst, uint8_t value) {
    if (dst.length == 0)
        return;
    Command c;
    c.kind = Command::Kind::Fill;
    c.destination = std::move(dst);
    c.fill_value = value;
    enqueue(std::move(c));
}
BufferView Stream::current_predicate() const {
    std::lock_guard<std::recursive_mutex> order(state_->domain->mutex);
    std::lock_guard<std::recursive_mutex> l(state_->mutex);
    return state_->predicate;
}
bool Stream::is_capturing() const {
    std::lock_guard<std::recursive_mutex> order(state_->domain->mutex);
    std::lock_guard<std::recursive_mutex> l(state_->mutex);
    return state_->recording == detail::StreamState::Recording::Capture;
}
bool Stream::is_recording() const {
    std::lock_guard<std::recursive_mutex> order(state_->domain->mutex);
    std::lock_guard<std::recursive_mutex> l(state_->mutex);
    return state_->recording != detail::StreamState::Recording::None;
}
void Stream::synchronize() {
    wait_stream(state_);
}
void Stream::check_errors() {
    std::lock_guard<std::recursive_mutex> order(state_->domain->mutex);
    std::lock_guard<std::recursive_mutex> l(state_->mutex);
    state_->check_thread();
    state_->check_error();
}
Ticket Stream::last_ticket() const {
    std::lock_guard<std::recursive_mutex> order(state_->domain->mutex);
    std::lock_guard<std::recursive_mutex> l(state_->mutex);
    return state_->last;
}
Ticket Stream::completion_ticket() {
    std::lock_guard<std::recursive_mutex> order(state_->domain->mutex);
    std::lock_guard<std::recursive_mutex> lock(state_->mutex);
    state_->check_thread();
    state_->check_error();
    require(state_->recording == detail::StreamState::Recording::None,
            "cannot query completion during recording");
    state_->submit({}, nullptr, {}, true);
    return state_->last;
}
Statistics Stream::statistics() const {
    std::lock_guard<std::recursive_mutex> order(state_->domain->mutex);
    std::lock_guard<std::recursive_mutex> l(state_->mutex);
    return state_->stats;
}
void Stream::reset_statistics() {
    std::lock_guard<std::recursive_mutex> order(state_->domain->mutex);
    std::lock_guard<std::recursive_mutex> l(state_->mutex);
    state_->check_thread();
    require(!state_->timing, "cannot reset statistics inside a timing window");
    state_->stats = {};
}
void Stream::begin_gpu_timing() {
    auto timing = std::make_shared<detail::TimingState>();
    timing->result.supported = state_->device.capabilities().backend == BackendKind::Metal;
    {
        std::lock_guard<std::recursive_mutex> order(state_->domain->mutex);
        std::lock_guard<std::recursive_mutex> lock(state_->mutex);
        state_->check_thread();
        state_->check_error();
        require(state_->recording == detail::StreamState::Recording::None,
                "finish recording before profiling");
        require(!state_->timing, "GPU timing is already active");
        require(!state_->active_synchronizers,
                "another thread is synchronizing this stream; join it before profiling");
        // Reserve ownership before the drain. A concurrent synchronize() may
        // have moved tickets out of outstanding; the count above detects it.
        state_->timing = std::move(timing);
        state_->timing_thread = std::this_thread::get_id();
    }
    try {
        synchronize(); // Neither old work nor an initial ordering marker is measured.
        std::lock_guard<std::recursive_mutex> order(state_->domain->mutex);
        std::lock_guard<std::recursive_mutex> lock(state_->mutex);
        state_->timing_collecting = true;
    } catch (...) {
        std::lock_guard<std::recursive_mutex> order(state_->domain->mutex);
        std::lock_guard<std::recursive_mutex> lock(state_->mutex);
        state_->timing.reset();
        state_->timing_thread = {};
        state_->timing_collecting = false;
        throw;
    }
}
GpuTiming Stream::end_gpu_timing() {
    std::shared_ptr<detail::TimingState> timing;
    {
        std::lock_guard<std::recursive_mutex> order(state_->domain->mutex);
        std::lock_guard<std::recursive_mutex> lock(state_->mutex);
        state_->check_thread();
        require(state_->timing != nullptr, "GPU timing is not active");
        require(state_->recording == detail::StreamState::Recording::None,
                "finish recording before ending GPU timing");
        timing = state_->timing;
    }
    std::exception_ptr error;
    try {
        synchronize();
    } catch (...) {
        error = std::current_exception();
    }
    {
        std::lock_guard<std::recursive_mutex> order(state_->domain->mutex);
        std::lock_guard<std::recursive_mutex> lock(state_->mutex);
        state_->timing.reset();
        state_->timing_thread = {};
        state_->timing_collecting = false;
    }
    if (error)
        std::rethrow_exception(error);
    return timing->snapshot();
}
EventPoint Stream::signal(TimelineEvent& event) {
    auto s = state_;
    std::lock_guard<std::recursive_mutex> order(s->domain->mutex);
    std::lock_guard<std::recursive_mutex> l(s->mutex);
    s->check_error();
    s->check_thread();
    require(s->recording == detail::StreamState::Recording::None,
            "event signals are submission-time operations; finish recording first");
    require(event.state_ && event.state_->device_identity == s->device.implementation().get(),
            "event and producer stream belong to different devices");
    std::lock_guard<std::mutex> reservation(event.state_->reservation_mutex);
    require(event.state_->producer_stream == 0 || event.state_->producer_stream == s->id,
            "a timeline event has one producer stream; use a distinct event for another producer");
    require(event.state_->next_value < UINT64_MAX, "timeline event value overflow");
    const auto value = ++event.state_->next_value;
    event.state_->producer_stream = s->id;
    detail::SubmitOptions options;
    options.signal_event = event.state_;
    options.signal_value = value;
    s->submit({}, nullptr, options);
    EventPoint point;
    point.state_ = event.state_;
    point.value_ = value;
    // The GPU timeline establishes ordering. The prefix also carries failures in
    // earlier submissions to host observers and downstream completion objects.
    point.completion_ = Ticket{detail::make_completion<detail::PrefixCompletion>(s->outstanding)};
    point.pending_submission_ = s->last_submission;
    return point;
}
Ticket Stream::wait(const EventPoint& point) {
    auto s = state_;
    std::lock_guard<std::recursive_mutex> order(s->domain->mutex);
    std::lock_guard<std::recursive_mutex> l(s->mutex);
    s->check_error();
    s->check_thread();
    require(s->recording == detail::StreamState::Recording::None,
            "event waits are submission-time operations; use Graph::launch(stream, waits)");
    detail::SubmitOptions options;
    options.waits = {point};
    s->submit({}, nullptr, options);
    return s->last;
}
void Stream::host(const std::function<void()>& fn) {
    {
        std::lock_guard<std::recursive_mutex> order(state_->domain->mutex);
        std::lock_guard<std::recursive_mutex> l(state_->mutex);
        state_->check_thread();
        state_->check_error();
        if (state_->recording == detail::StreamState::Recording::Capture)
            throw std::logic_error("host callbacks cannot be replayed by Metal; split the graph at "
                                   "the CPU dependency");
        require(state_->nested_scopes.empty() && !state_->parallel_active && !state_->predicate,
                "host boundaries cannot occur inside recording transactions or parallel/GPU "
                "control-flow groups");
        auto commands = std::move(state_->pending);
        state_->pending.clear();
        if (!state_->pending_ordered.empty())
            state_->commit_ordered();
        else
            state_->submit(commands, nullptr, {}, true);
    }
    wait_stream(state_, true);
    fn();
}
Stream& default_stream() {
    thread_local std::vector<std::unique_ptr<Stream>> defaults;
    auto d = Device::current();
    for (auto& s : defaults)
        if (s->device() == d)
            return *s;
    defaults.push_back(std::make_unique<Stream>(d));
    return *defaults.back();
}
Stream& ordered_default_stream(Device device) {
    static std::mutex mutex;
    static std::vector<std::unique_ptr<Stream>> defaults;
    std::lock_guard<std::mutex> lock(mutex);
    for (auto& stream : defaults)
        if (stream->device() == device)
            return *stream;
    defaults.push_back(
        std::make_unique<Stream>(device, DispatchPolicy::Direct, StreamOrdering::Default));
    return *defaults.back();
}
Stream& stream_or_default(Stream* s) {
    return s ? *s : default_stream();
}
void synchronize_device(Device d) {
    std::vector<std::shared_ptr<detail::StreamState>> live;
    {
        std::lock_guard<std::mutex> l(streams_mutex);
        for (auto i = streams.begin(); i != streams.end();) {
            if (auto s = i->lock()) {
                if (s->device == d)
                    live.push_back(std::move(s));
                ++i;
            } else
                i = streams.erase(i);
        }
    }
    // Reject before waiting any queue; avoids partial synchronization in a capture.
    for (auto& s : live) {
        std::lock_guard<std::recursive_mutex> order(s->domain->mutex);
        std::lock_guard<std::recursive_mutex> l(s->mutex);
        if (s->recording != detail::StreamState::Recording::None)
            throw std::logic_error(
                "device synchronization is not allowed while a stream is recording");
    }
    std::exception_ptr error;
    for (auto& s : live)
        try {
            wait_stream(s);
        } catch (...) {
            if (!error)
                error = std::current_exception();
        }
    if (error)
        std::rethrow_exception(error);
}
void synchronize_buffer(BufferView buffer) {
    if (!buffer.length)
        return;
    require(static_cast<bool>(buffer), "cannot synchronize an empty allocation");
    require(buffer.offset <= buffer.buffer.size() &&
                buffer.length <= buffer.buffer.size() - buffer.offset,
            "invalid host-access buffer span");
    if (buffer.buffer.implementation()->fresh_host_storage.load(std::memory_order_acquire))
        return;
    synchronize_device(buffer.buffer.device());
}

} // namespace metalrt
