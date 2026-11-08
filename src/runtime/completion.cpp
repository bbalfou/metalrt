// Submission completion, timing accumulation, and exception-safe scope lifetime.
#include "runtime_internal.h"

namespace metalrt {
Statistics& Statistics::operator+=(const Statistics& s) {
    command_buffers += s.command_buffers;
    compute_dispatches += s.compute_dispatches;
    blit_commands += s.blit_commands;
    icb_builds += s.icb_builds;
    icb_reuses += s.icb_reuses;
    icb_updates += s.icb_updates;
    control_dispatches += s.control_dispatches;
    event_waits += s.event_waits;
    event_signals += s.event_signals;
    prepare_calls += s.prepare_calls;
    cpu_submit_nanoseconds += s.cpu_submit_nanoseconds;
    stream_wait_calls += s.stream_wait_calls;
    cpu_wait_nanoseconds += s.cpu_wait_nanoseconds;
    return *this;
}
void detail::TimingState::record(double begin, double end, bool successful) {
    std::lock_guard<std::mutex> lock(mutex);
    ++result.command_buffers;
    if (!successful || !std::isfinite(begin) || !std::isfinite(end) || begin <= 0 || end <= begin)
        return;
    if (result.timed_command_buffers == 0) {
        result.first_gpu_start_seconds = begin;
        result.last_gpu_end_seconds = end;
    } else {
        result.first_gpu_start_seconds = std::min(result.first_gpu_start_seconds, begin);
        result.last_gpu_end_seconds = std::max(result.last_gpu_end_seconds, end);
    }
    intervals.emplace_back(begin, end);
    ++result.timed_command_buffers;
    result.command_buffer_seconds += end - begin;
}
GpuTiming detail::TimingState::snapshot() const {
    std::lock_guard<std::mutex> lock(mutex);
    auto snapshot = result;
    auto sorted = intervals;
    std::sort(sorted.begin(), sorted.end());
    double begin = 0, end = 0;
    for (const auto& interval : sorted) {
        if (interval.first > end) {
            snapshot.occupied_seconds += end - begin;
            begin = interval.first;
            end = interval.second;
        } else
            end = std::max(end, interval.second);
    }
    snapshot.occupied_seconds += end - begin;
    return snapshot;
}
TimelineEvent::TimelineEvent(Device device) {
    require(static_cast<bool>(device), "empty event device");
    state_ = device.implementation()->make_event();
    if (!state_)
        throw std::runtime_error("backend did not create a timeline event");
    state_->device_owner = device.implementation();
    state_->device_identity = device.implementation().get();
}
uint64_t TimelineEvent::completed_value() const {
    return state_->completed_value();
}
Device TimelineEvent::device() const {
    return Device{state_->device_owner};
}
void detail::destroy_completion(Completion* value) noexcept {
    thread_local Completion* pending = nullptr;
    thread_local bool draining = false;
    value->deletion_next = pending;
    pending = value;
    if (draining)
        return;
    draining = true;
    while (pending) {
        auto* next = pending;
        pending = next->deletion_next;
        delete next;
    }
    draining = false;
}
void Ticket::wait() const {
    if (!completion_)
        return;
    struct Visit {
        std::shared_ptr<detail::Completion> completion;
        bool observed = false;
    };
    std::vector<Visit> pending{{completion_, false}};
    std::unordered_set<detail::Completion*> seen;
    std::exception_ptr first;
    while (!pending.empty()) {
        auto current = std::move(pending.back());
        pending.pop_back();
        try {
            if (current.observed) {
                current.completion->wait();
                continue;
            }
            if (!seen.insert(current.completion.get()).second)
                continue;
            std::vector<Ticket> dependencies;
            if (!current.completion->append_dependencies(dependencies, true))
                throw std::logic_error("completion did not attach its waiting dependencies");
            pending.push_back({current.completion, true});
            for (auto i = dependencies.rbegin(); i != dependencies.rend(); ++i)
                if (i->completion_)
                    pending.push_back({std::move(i->completion_), false});
        } catch (const UncommittedCommandBufferError&) {
            // A later committed buffer can be GPU-blocked on this producer's
            // event. Draining that dependent before its owner can commit would
            // deadlock; leave the entire ticket DAG intact for a later retry.
            throw;
        } catch (...) {
            if (!first)
                first = std::current_exception();
        }
    }
    if (first)
        std::rethrow_exception(first);
}
bool Ticket::ready() const {
    if (!completion_)
        return true;
    std::vector<std::shared_ptr<detail::Completion>> pending{completion_};
    std::unordered_set<detail::Completion*> seen;
    while (!pending.empty()) {
        auto current = std::move(pending.back());
        pending.pop_back();
        if (!seen.insert(current.get()).second)
            continue;
        if (!current->ready())
            return false;
        std::vector<Ticket> dependencies;
        if (!current->append_dependencies(dependencies, false))
            return false;
        for (auto& dependency : dependencies)
            if (dependency.completion_)
                pending.push_back(std::move(dependency.completion_));
    }
    return true;
}

Scope::Scope(std::function<void()> f, std::function<void()> c,
             std::function<void(std::exception_ptr)> e)
    : finish_(std::move(f)), cancel_(std::move(c)), error_(std::move(e)),
      exceptions_(std::uncaught_exceptions()), active_(true) {}
Scope::Scope(Scope&& s) noexcept
    : finish_(std::move(s.finish_)), cancel_(std::move(s.cancel_)), error_(std::move(s.error_)),
      exceptions_(s.exceptions_), active_(s.active_) {
    s.active_ = false;
}
Scope::~Scope() noexcept {
    if (!active_)
        return;
    if (std::uncaught_exceptions() > exceptions_) {
        cancel();
        return;
    }
    try {
        finish();
    } catch (...) {
        if (error_)
            error_(std::current_exception());
    }
}
void Scope::finish() {
    if (!active_)
        return;
    active_ = false;
    try {
        if (finish_)
            finish_();
    } catch (...) {
        if (cancel_)
            cancel_();
        throw;
    }
}
void Scope::cancel() noexcept {
    if (!active_)
        return;
    active_ = false;
    try {
        if (cancel_)
            cancel_();
    } catch (...) {
        if (error_)
            error_(std::current_exception());
    }
}

} // namespace metalrt
