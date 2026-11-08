// Batch, transaction, parallel, and predicate recording scopes.
#include "runtime_internal.h"

namespace metalrt {
Scope Stream::batch() {
    auto s = state_;
    uint64_t token;
    {
        std::lock_guard<std::recursive_mutex> order(s->domain->mutex);
        std::lock_guard<std::recursive_mutex> l(s->mutex);
        s->check_error();
        s->check_thread();
        if (s->recording != detail::StreamState::Recording::None)
            return {};
        s->recording = detail::StreamState::Recording::Batch;
        s->recording_thread = std::this_thread::get_id();
        token = ++s->generation;
    }
    auto cancel = [s, token] {
        std::lock_guard<std::recursive_mutex> order(s->domain->mutex);
        std::lock_guard<std::recursive_mutex> l(s->mutex);
        if (s->generation == token) {
            s->cancel_ordered(0);
            s->pending.clear();
            end_recording(*s);
        }
    };
    try {
        return Scope{[s, token] {
                         std::lock_guard<std::recursive_mutex> order(s->domain->mutex);
                         std::lock_guard<std::recursive_mutex> l(s->mutex);
                         s->check_thread();
                         require(s->generation == token, "scope finished out of order");
                         require(s->nested_scopes.empty() && !s->parallel_active && !s->predicate,
                                 "finish nested recording scopes before the enclosing batch");
                         auto commands = std::move(s->pending);
                         s->pending.clear();
                         end_recording(*s);
                         if (!s->pending_ordered.empty())
                             s->commit_ordered();
                         else
                             s->submit(commands);
                     },
                     cancel, [s](std::exception_ptr e) { defer_error(s, e); }};
    } catch (...) {
        cancel();
        throw;
    }
}
Scope Stream::transaction() {
    auto outer = std::make_shared<Scope>(batch());
    auto s = state_;
    RecordingCheckpoint saved;
    {
        std::lock_guard<std::recursive_mutex> order(s->domain->mutex);
        std::lock_guard<std::recursive_mutex> l(s->mutex);
        s->check_error();
        s->check_thread();
        saved = begin_nested_scope(*s);
    }
    auto cancel = [s, outer, saved] {
        {
            std::lock_guard<std::recursive_mutex> order(s->domain->mutex);
            std::lock_guard<std::recursive_mutex> l(s->mutex);
            cancel_nested_scope(*s, saved);
        }
        outer->cancel();
    };
    try {
        return Scope{[s, outer, saved] {
                         {
                             std::lock_guard<std::recursive_mutex> order(s->domain->mutex);
                             std::lock_guard<std::recursive_mutex> l(s->mutex);
                             s->check_thread();
                             if (!finish_nested_scope(*s, saved))
                                 return;
                         }
                         outer->finish();
                     },
                     cancel, [s](std::exception_ptr e) { defer_error(s, e); }};
    } catch (...) {
        cancel();
        throw;
    }
}
Scope Stream::parallel() {
    auto outer = std::make_shared<Scope>(batch());
    auto s = state_;
    RecordingCheckpoint saved;
    {
        std::lock_guard<std::recursive_mutex> order(s->domain->mutex);
        std::lock_guard<std::recursive_mutex> l(s->mutex);
        s->check_thread();
        s->check_error();
        require(!s->parallel_active && !s->predicate,
                "parallel groups cannot be nested or predicated");
        require(s->next_parallel_id < UINT64_MAX, "parallel group identifier overflow");
        saved = begin_nested_scope(*s);
        s->parallel_active = true;
        s->parallel_first = true;
        s->parallel_id = ++s->next_parallel_id;
    }
    auto cancel = [s, outer, saved] {
        {
            std::lock_guard<std::recursive_mutex> order(s->domain->mutex);
            std::lock_guard<std::recursive_mutex> l(s->mutex);
            cancel_nested_scope(*s, saved);
        }
        outer->cancel();
    };
    try {
        return Scope{[s, outer, saved] {
                         {
                             std::lock_guard<std::recursive_mutex> order(s->domain->mutex);
                             std::lock_guard<std::recursive_mutex> l(s->mutex);
                             s->check_thread();
                             if (!finish_nested_scope(*s, saved))
                                 return;
                             s->parallel_active = saved.parallel_active;
                             s->parallel_first = saved.parallel_first;
                             s->parallel_id = saved.parallel_id;
                             s->force_join =
                                 s->pending.size() > saved.pending ? true : saved.force_join;
                         }
                         outer->finish();
                     },
                     cancel, [s](std::exception_ptr e) { defer_error(s, e); }};
    } catch (...) {
        cancel();
        throw;
    }
}
Scope Stream::predicate_scope(BufferView predicate) {
    validate_buffer(predicate, device());
    require(predicate.offset % 4 == 0 && predicate.length >= 4,
            "predicate scope requires an aligned uint32 value");
    auto outer = std::make_shared<Scope>(batch());
    auto s = state_;
    RecordingCheckpoint saved;
    {
        std::lock_guard<std::recursive_mutex> order(s->domain->mutex);
        std::lock_guard<std::recursive_mutex> l(s->mutex);
        s->check_thread();
        s->check_error();
        require(!s->parallel_active, "predicate scope cannot be placed inside a parallel group");
        saved = begin_nested_scope(*s);
        s->predicate = std::move(predicate);
    }
    auto cancel = [s, outer, saved] {
        {
            std::lock_guard<std::recursive_mutex> order(s->domain->mutex);
            std::lock_guard<std::recursive_mutex> l(s->mutex);
            cancel_nested_scope(*s, saved);
        }
        outer->cancel();
    };
    try {
        return Scope{[s, outer, saved] {
                         {
                             std::lock_guard<std::recursive_mutex> order(s->domain->mutex);
                             std::lock_guard<std::recursive_mutex> l(s->mutex);
                             s->check_thread();
                             if (!finish_nested_scope(*s, saved))
                                 return;
                             s->predicate = saved.predicate;
                         }
                         outer->finish();
                     },
                     cancel, [s](std::exception_ptr e) { defer_error(s, e); }};
    } catch (...) {
        cancel();
        throw;
    }
}

} // namespace metalrt
