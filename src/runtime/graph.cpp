// Capture publication, plan preparation, and replay.
#include "runtime_internal.h"

namespace metalrt {
Graph::Graph() : state_(std::make_shared<detail::GraphState>()) {}
Graph::~Graph() = default;
Scope Graph::capture_guard(Stream* stream) {
    return record(stream, true);
}
Scope Graph::record_guard(Stream* stream) {
    return record(stream, false);
}
Scope Graph::record(Stream* stream, bool execute) {
    auto s = stream_or_default(stream).state_;
    auto g = state_;
    uint64_t token;
    {
        std::lock_guard<std::recursive_mutex> order(s->domain->mutex);
        std::lock_guard<std::recursive_mutex> l(s->mutex);
        s->check_error();
        s->check_thread();
        if (s->recording != detail::StreamState::Recording::None)
            return {};
        std::lock_guard<std::mutex> gl(g->mutex);
        require(!g->capturing, "graph is already being captured on another stream");
        g->capturing = true;
        s->recording = detail::StreamState::Recording::Capture;
        s->recording_thread = std::this_thread::get_id();
        token = ++s->generation;
        g->capture_owner = s->id;
        g->capture_generation = token;
    }
    auto cancel = [s, g, token] {
        std::lock_guard<std::recursive_mutex> order(s->domain->mutex);
        std::lock_guard<std::recursive_mutex> l(s->mutex);
        if (s->generation == token) {
            s->cancel_ordered(0);
            s->pending.clear();
            end_recording(*s);
        }
        std::lock_guard<std::mutex> gl(g->mutex);
        if (g->capture_owner == s->id && g->capture_generation == token)
            g->capturing = false;
    };
    try {
        return Scope{[s, g, token, execute] {
                         bool sync = false;
                         {
                             std::lock_guard<std::recursive_mutex> order(s->domain->mutex);
                             std::lock_guard<std::recursive_mutex> l(s->mutex);
                             s->check_thread();
                             require(s->generation == token, "capture finished out of order");
                             require(
                                 s->nested_scopes.empty() && !s->parallel_active && !s->predicate,
                                 "finish nested recording scopes before ending graph recording");
                             auto commands = std::move(s->pending);
                             s->pending.clear();
                             end_recording(*s);
                             std::lock_guard<std::mutex> gl(g->mutex);
                             auto plan = (g->device && g->device != s->device)
                                             ? std::shared_ptr<detail::Plan>{}
                                             : g->plan;
                             g->capturing = false;
                             // Publish the new graph only after submission succeeds. A failed
                             // backend update may have touched an idle plan, so discard that cache
                             // on error.
                             try {
                                 if (commands.empty())
                                     plan.reset();
                                 else if (execute)
                                     s->submit(commands, &plan);
                             } catch (...) {
                                 g->plan.reset();
                                 throw;
                             }
                             g->commands = std::move(commands);
                             g->plan = std::move(plan);
                             g->device = s->device;
                             ++g->revision;
                             if (execute)
                                 sync = std::exchange(g->synchronize, false);
                         }
                         if (sync)
                             wait_stream(s);
                     },
                     cancel, [s](std::exception_ptr e) { defer_error(s, e); }};
    } catch (...) {
        cancel();
        throw;
    }
}
void Graph::prepare(Stream* stream) {
    auto s = stream_or_default(stream).state_;
    auto g = state_;
    std::lock_guard<std::recursive_mutex> order(s->domain->mutex);
    std::lock_guard<std::recursive_mutex> l(s->mutex);
    s->check_error();
    s->check_thread();
    require(s->recording == detail::StreamState::Recording::None,
            "cannot prepare a graph while a stream is recording");
    std::lock_guard<std::mutex> gl(g->mutex);
    require(!g->capturing, "cannot prepare a recording graph");
    if (g->commands.empty())
        return;
    require(g->device == s->device, "graph and preparing stream belong to different devices");
    detail::SubmitOptions options;
    options.prepare_only = true;
    s->submit(g->commands, &g->plan, options);
}
Ticket Graph::launch(Stream* stream, const std::vector<EventPoint>& waits) {
    auto s = stream_or_default(stream).state_;
    auto g = state_;
    std::lock_guard<std::recursive_mutex> order(s->domain->mutex);
    std::lock_guard<std::recursive_mutex> l(s->mutex);
    s->check_error();
    s->check_thread();
    std::lock_guard<std::mutex> gl(g->mutex);
    require(!g->capturing, "cannot replay a graph while it is being recorded");
    if (g->commands.empty() && waits.empty())
        return {};
    require(!g->device || g->device == s->device,
            "graph and replay stream belong to different devices");
    if (s->recording != detail::StreamState::Recording::None) {
        require(waits.empty(), "event waits cannot be frozen into a replayable capture");
        auto saved = checkpoint(*s);
        try {
            std::map<uint64_t, uint64_t> groups;
            for (auto command : g->commands) {
                if (command.kind == Command::Kind::Compute && command.compute.parallel_group &&
                    !s->parallel_active) {
                    auto old = command.compute.parallel_group;
                    if (!groups.count(old)) {
                        require(s->next_parallel_id < UINT64_MAX,
                                "parallel group identifier overflow");
                        groups[old] = ++s->next_parallel_id;
                    }
                    command.compute.parallel_group = groups[old];
                }
                append_command(*s, std::move(command));
            }
        } catch (...) {
            restore_checkpoint(*s, saved);
            throw;
        }
        return {};
    }
    detail::SubmitOptions options;
    options.waits = waits;
    s->submit(g->commands, &g->plan, options);
    return s->last;
}
Graph Graph::clone() const {
    std::lock_guard<std::mutex> l(state_->mutex);
    require(!state_->capturing, "cannot clone a recording graph");
    // Independent graph variants need independent mutable plan caches. Sharing
    // an idle Plan would be numerically safe after comparison, but preparing or
    // rotating one variant would re-encode the other variant's cached bindings.
    Graph result;
    result.state_->commands = state_->commands;
    result.state_->device = state_->device;
    return result;
}
uint64_t Graph::revision() const {
    std::lock_guard<std::mutex> l(state_->mutex);
    return state_->revision;
}
Graph::Update Graph::update() {
    std::lock_guard<std::mutex> l(state_->mutex);
    require(!state_->capturing, "cannot update a recording graph");
    return Update{state_, state_->revision, state_->commands, state_->device};
}
void Graph::reset() {
    std::lock_guard<std::mutex> l(state_->mutex);
    require(!state_->capturing, "cannot reset a capturing graph");
    state_->commands.clear();
    state_->plan.reset();
    state_->device = {};
    state_->synchronize = false;
    ++state_->revision;
}
void Graph::schedule_synchronize() {
    std::lock_guard<std::mutex> l(state_->mutex);
    state_->synchronize = true;
}
size_t Graph::size() const {
    std::lock_guard<std::mutex> l(state_->mutex);
    return state_->commands.size();
}
std::vector<Command> Graph::commands() const {
    std::lock_guard<std::mutex> l(state_->mutex);
    return state_->commands;
}

} // namespace metalrt
