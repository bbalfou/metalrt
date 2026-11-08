// Queue ordering and deferred submission behind cancellable recordings.
#include "runtime_internal.h"

namespace metalrt {
namespace detail {
std::atomic<uint64_t> next_stream_id{1};
std::shared_ptr<OrderingDomain> ordering_domain(const Device& device) {
    static std::mutex mutex;
    static std::map<const void*, std::weak_ptr<OrderingDomain>> domains;
    std::lock_guard<std::mutex> lock(mutex);
    auto& entry = domains[device.implementation().get()];
    auto domain = entry.lock();
    if (!domain) {
        domain = std::make_shared<OrderingDomain>();
        entry = domain;
    }
    return domain;
}
bool blocked_by_recording(const std::shared_ptr<OrderedSubmission>& node, std::thread::id thread) {
    std::vector<std::shared_ptr<OrderedSubmission>> pending{node};
    std::unordered_set<OrderedSubmission*> seen;
    while (!pending.empty()) {
        auto current = std::move(pending.back());
        pending.pop_back();
        if (current->submitted || !seen.insert(current.get()).second)
            continue;
        if (!current->committed && current->recording_thread == thread)
            return true;
        pending.insert(pending.end(), current->dependencies.begin(), current->dependencies.end());
    }
    return false;
}
// All queue mutations take domain, then stream locks. This permits the scheduler
// to submit another queue's committed segment without reversing the lock order.
void drain_ordered(const std::shared_ptr<OrderingDomain>& domain) {
    bool progress = true;
    while (progress) {
        progress = false;
        for (size_t index = 0; index < domain->pending.size(); ++index) {
            auto node = domain->pending[index];
            if (node->submitted || !node->committed ||
                std::any_of(node->dependencies.begin(), node->dependencies.end(),
                            [](const auto& dependency) { return !dependency->submitted; }))
                continue;
            auto owner = node->owner;
            std::lock_guard<std::recursive_mutex> lock(owner->mutex);
            for (const auto& dependency : node->dependencies)
                if (dependency->failure && !node->failure)
                    node->failure = dependency->failure;
            try {
                if (!node->failure &&
                    !(node->cancelled && !node->observed && node->completion->prefix.ready())) {
                    auto result = owner->backend_submit(node->commands, node->options, node->policy,
                                                        node->plan);
                    node->plan = std::move(result.plan);
                    node->completion->resolve(std::move(result.completion));
                } else
                    node->completion->resolve({}, node->failure);
            } catch (...) {
                node->failure = std::current_exception();
                node->plan.reset();
                node->completion->resolve({}, node->failure);
            }
            node->submitted = true;
            node->commands.clear();
            node->dependencies.clear();
            node->owner.reset();
            progress = true;
        }
        domain->pending.erase(std::remove_if(domain->pending.begin(), domain->pending.end(),
                                             [](const auto& node) { return node->submitted; }),
                              domain->pending.end());
    }
}
std::shared_ptr<OrderedSubmission> StreamState::reserve_ordered(std::vector<Command> commands,
                                                                const SubmitOptions& options,
                                                                bool committed,
                                                                std::shared_ptr<Plan> cached) {
    outstanding.reserve(1);
    auto seen = ordering_seen;
    auto node = std::make_shared<OrderedSubmission>();
    node->owner = shared_from_this();
    node->commands = std::move(commands);
    node->options = options;
    node->options.timing = timing_collecting ? timing : nullptr;
    node->policy = policy;
    node->plan = std::move(cached);
    node->committed = committed;
    node->recording_thread = recording_thread;
    node->completion = make_completion<DeferredCompletion>();
    const auto add_dependency = [&](const std::shared_ptr<OrderedSubmission>& dependency,
                                    bool cross_queue) {
        if (!dependency)
            return;
        if (dependency->submitted && dependency->point.completion().ready()) {
            dependency->point.wait();
            return;
        }
        if (std::find(node->dependencies.begin(), node->dependencies.end(), dependency) ==
            node->dependencies.end())
            node->dependencies.push_back(dependency);
        if (cross_queue) {
            dependency->observed = true;
            node->options.waits.push_back(dependency->point);
        }
    };
    add_dependency(ordering_tail, false);
    if (ordering_tail && !ordering_tail->point.completion().ready())
        node->completion->prefix = ordering_tail->point.completion();
    else if (!ordering_tail && !outstanding.empty())
        node->completion->prefix = Ticket{make_completion<PrefixCompletion>(outstanding)};
    for (const auto& point : options.waits) {
        auto dependency = point.pending_submission_.lock();
        if (dependency) {
            dependency->observed = true;
            add_dependency(dependency, false);
        }
    }
    if (ordered())
        for (auto i = domain->streams.begin(); i != domain->streams.end();) {
            auto other = i->lock();
            if (!other) {
                i = domain->streams.erase(i);
                continue;
            }
            ++i;
            if (other.get() == this || !((ordering == StreamOrdering::Default &&
                                          other->ordering == StreamOrdering::Blocking) ||
                                         (ordering == StreamOrdering::Blocking &&
                                          other->ordering == StreamOrdering::Default)))
                continue;
            // A reusable capture does not issue work, and cannot become a dependency
            // of the implicit default queue. Independent/nonblocking captures are fine.
            require(other->recording != Recording::Capture,
                    "implicit default-stream ordering cannot depend on an active reusable capture");
            add_dependency(other->ordering_tail, true);
            if (other->ordering_tail)
                seen[other->id] = other->ordering_tail->point.value();
        }
    if (!ordering_event) {
        ordering_event = device.implementation()->make_event();
        require(static_cast<bool>(ordering_event), "backend did not create an ordering event");
        ordering_event->device_identity = device.implementation().get();
        ordering_event->device_owner = device.implementation();
    }
    require(ordering_value < UINT64_MAX, "default-stream ordering event value overflow");
    node->point.state_ = ordering_event;
    node->point.value_ = ++ordering_value;
    node->point.completion_ = Ticket{node->completion};
    node->point.pending_submission_ = node;
    node->options.signals.push_back({ordering_event, ordering_value});
    std::weak_ptr<OrderedSubmission> weak_node = node;
    std::weak_ptr<OrderingDomain> weak_domain = domain;
    node->completion->recording_dependency = [weak_node, weak_domain] {
        auto domain = weak_domain.lock();
        auto node = weak_node.lock();
        if (!domain || !node)
            return false;
        std::lock_guard<std::recursive_mutex> lock(domain->mutex);
        return blocked_by_recording(node, std::this_thread::get_id());
    };
    domain->pending.push_back(node);
    ordering_seen.swap(seen);
    ordering_tail = node;
    last_submission = node;
    domain->last_issued = node;
    return node;
}
void StreamState::submit(const std::vector<Command>& commands, std::shared_ptr<Plan>* cache,
                         const SubmitOptions& options, bool boundary) {
    check_thread();
    const bool empty = commands.empty() && options.waits.empty() && !options.signal_event &&
                       options.signals.empty();
    if (empty && !options.prepare_only && !(boundary && ordered()))
        return;
    if (empty && boundary && ordered()) {
        bool new_dependency = false;
        for (const auto& weak : domain->streams)
            if (auto other = weak.lock()) {
                if (other.get() == this || !((ordering == StreamOrdering::Default &&
                                              other->ordering == StreamOrdering::Blocking) ||
                                             (ordering == StreamOrdering::Blocking &&
                                              other->ordering == StreamOrdering::Default)))
                    continue;
                require(
                    other->recording != Recording::Capture,
                    "implicit default-stream ordering cannot depend on an active reusable capture");
                if (other->ordering_tail &&
                    ordering_seen[other->id] < other->ordering_tail->point.value())
                    new_dependency = true;
            }
        if (!new_dependency)
            return;
    }
    bool deferred = ordering_tail && !ordering_tail->submitted;
    for (const auto& point : options.waits)
        if (point.pending_submission_.lock())
            deferred = true;
    if (!options.prepare_only && (ordered() || deferred)) {
        auto node = reserve_ordered(commands, options, true, cache ? *cache : nullptr);
        publish_ordered();
        if (std::all_of(node->dependencies.begin(), node->dependencies.end(),
                        [](const auto& dependency) { return dependency->submitted; }))
            drain_ordered(domain);
        if (cache && node->submitted)
            *cache = node->plan;
        if (node->failure) {
            if (cache)
                cache->reset();
            std::rethrow_exception(node->failure);
        }
        return;
    }
    auto profiled_options = options;
    profiled_options.timing = timing_collecting ? timing : nullptr;
    SubmitResult result;
    try {
        result = backend_submit(commands, profiled_options, policy, cache ? *cache : nullptr);
    } catch (...) {
        if (cache)
            cache->reset();
        throw;
    }
    if (cache)
        *cache = result.plan;
    if (result.completion) {
        last_submission.reset();
        // An independent queue can resume direct submission once an earlier
        // deferred prefix has been attached. A later enrollment must snapshot
        // this entire current prefix, including these independent submissions.
        if (!ordered())
            ordering_tail.reset();
        last = Ticket{std::move(result.completion)};
        outstanding.push_back(last);
    }
}
void StreamState::record_ordered(const Command& command) {
    auto tail = domain->last_issued.lock();
    if (tail && !pending_ordered.empty() && tail == pending_ordered.back() && !tail->committed) {
        tail->commands.push_back(command);
        return;
    }
    pending_ordered.reserve(checked_add(pending_ordered.size(), 1));
    auto node = reserve_ordered({command}, {}, false);
    node->pending_begin = pending.size() - 1;
    pending_ordered.push_back(std::move(node));
}
void StreamState::commit_ordered() {
    auto nodes = std::move(pending_ordered);
    pending_ordered.clear();
    outstanding.reserve(1);
    for (const auto& node : nodes)
        node->committed = true;
    publish_ordered();
    drain_ordered(domain);
    for (const auto& node : nodes)
        if (node->failure)
            std::rethrow_exception(node->failure);
}
void StreamState::cancel_ordered(size_t keep) {
    for (auto i = pending_ordered.begin(); i != pending_ordered.end();) {
        auto& node = *i;
        if (node->pending_begin >= keep) {
            node->commands.clear();
            node->cancelled = true;
            node->committed = true;
            i = pending_ordered.erase(i);
        } else {
            if (node->commands.size() > keep - node->pending_begin)
                node->commands.resize(keep - node->pending_begin);
            ++i;
        }
    }
    if (ordering_tail)
        publish_ordered();
    drain_ordered(domain);
}
} // namespace detail

} // namespace metalrt
