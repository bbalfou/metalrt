// Nested recording checkpoints and command accumulation.
#include "runtime_internal.h"

namespace metalrt {
RecordingCheckpoint checkpoint(const detail::StreamState& s) {
    return {s.pending.size(), s.generation,       0,
            s.parallel_id,    s.next_parallel_id, s.parallel_active,
            s.parallel_first, s.force_join,       s.predicate};
}
RecordingCheckpoint begin_nested_scope(detail::StreamState& s) {
    auto result = checkpoint(s);
    require(s.next_scope_id < UINT64_MAX, "recording scope identifier overflow");
    result.scope = ++s.next_scope_id;
    s.nested_scopes.push_back(result.scope);
    return result;
}
bool finish_nested_scope(detail::StreamState& s, const RecordingCheckpoint& saved) {
    if (s.generation != saved.generation || s.recording == detail::StreamState::Recording::None)
        return false;
    auto found = std::find(s.nested_scopes.begin(), s.nested_scopes.end(), saved.scope);
    if (found == s.nested_scopes.end())
        return false; // An ancestor already cancelled it.
    require(s.nested_scopes.back() == saved.scope,
            "recording scopes must finish in reverse creation order");
    s.nested_scopes.pop_back();
    return true;
}
void restore_checkpoint(detail::StreamState& s, const RecordingCheckpoint& saved) {
    if (s.pending.size() >= saved.pending) {
        s.cancel_ordered(saved.pending);
        s.pending.resize(saved.pending);
    }
    s.parallel_id = saved.parallel_id;
    s.next_parallel_id = saved.next_parallel_id;
    s.parallel_active = saved.parallel_active;
    s.parallel_first = saved.parallel_first;
    s.force_join = saved.force_join;
    s.predicate = saved.predicate;
}
void cancel_nested_scope(detail::StreamState& s, const RecordingCheckpoint& saved) {
    if (s.generation != saved.generation || s.recording == detail::StreamState::Recording::None)
        return;
    auto found = std::find(s.nested_scopes.begin(), s.nested_scopes.end(), saved.scope);
    if (found == s.nested_scopes.end())
        return;
    restore_checkpoint(s, saved);
    s.nested_scopes.erase(found, s.nested_scopes.end());
}
void end_recording(detail::StreamState& s) {
    s.recording = detail::StreamState::Recording::None;
    s.recording_thread = {};
    s.predicate = {};
    s.parallel_active = false;
    s.parallel_first = false;
    s.force_join = false;
    s.parallel_id = 0;
    s.nested_scopes.clear();
    ++s.generation;
}
void append_command(detail::StreamState& s, Command command) {
    if (s.predicate) {
        require(command.kind == Command::Kind::Compute,
                "GPU control-flow bodies accept compute commands only");
        if (!command.compute.control) {
            require(!command.compute.predicate,
                    "nested predicates must use structured if_else/repeat_while");
            command.compute.predicate = s.predicate;
        }
    }
    if (s.parallel_active) {
        require(command.kind == Command::Kind::Compute && !command.compute.control &&
                    !command.compute.indirect_groups && !command.compute.predicate,
                "parallel groups accept fixed, unpredicated compute commands only");
        command.compute.parallel_group = s.parallel_id;
        command.compute.barrier_before = s.parallel_first;
        for (auto i = s.pending.rbegin(); i != s.pending.rend(); ++i) {
            if (i->kind != Command::Kind::Compute || i->compute.parallel_group != s.parallel_id)
                break;
            validate_parallel_pair(i->compute, command.compute);
        }
    } else if (s.force_join && command.kind == Command::Kind::Compute) {
        command.compute.barrier_before = true;
    }
    const bool compute = command.kind == Command::Kind::Compute;
    validate_command(command, s.device);
    if (s.recording != detail::StreamState::Recording::None) {
        s.pending.push_back(command);
        try {
            if (s.recording == detail::StreamState::Recording::Batch &&
                (s.ordered() || !s.pending_ordered.empty()))
                s.record_ordered(command);
        } catch (...) {
            s.pending.pop_back();
            throw;
        }
    } else
        s.submit({std::move(command)});
    // Invalid/rejected commands must not consume the next dependency barrier.
    if (s.parallel_active)
        s.parallel_first = false;
    else if (s.force_join && compute)
        s.force_join = false;
}
void defer_error(const std::shared_ptr<detail::StreamState>& s, std::exception_ptr error) {
    std::lock_guard<std::recursive_mutex> order(s->domain->mutex);
    std::lock_guard<std::recursive_mutex> lock(s->mutex);
    if (!s->error)
        s->error = error;
}

} // namespace metalrt
