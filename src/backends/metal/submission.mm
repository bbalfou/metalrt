// Command-buffer submission, event ordering, and completion retention.
#include "queue.h"
namespace metalrt {
namespace detail {
namespace metal {
SubmitResult MetalQueue::submit(const std::vector<Command>& commands, DispatchPolicy policy,
                                std::shared_ptr<Plan> cached, const SubmitOptions& options) {
    std::lock_guard<std::mutex> queue_lock(mutex_);
    @autoreleasepool {
        if (policy == DispatchPolicy::Indirect && !indirect_supported_)
            throw std::runtime_error("Indirect policy requires compute indirect command buffer "
                                     "support on this Metal device");
        const bool needs_explicit_fences = validate(commands, policy);
        for (const auto& point : options.waits)
            native_event(point.implementation(), identity_);
        if (options.signal_event)
            native_event(options.signal_event, identity_);
        for (const auto& signal : options.signals)
            native_event(signal.event, identity_);
        if (options.prepare_only &&
            (!options.waits.empty() || options.signal_event || !options.signals.empty()))
            throw std::invalid_argument("preparing a plan cannot wait for or signal an event");
        SubmitResult result;
        auto completion = make_completion<MetalCompletion>();
        completion->dependencies = options.waits;
        completion->timing = options.timing;
        completion->externally_managed = externally_managed();
        result.completion = completion;
        std::unique_lock<std::mutex> plan_lock;
        auto plan = prepare_plan(commands, policy, std::move(cached), result.statistics, plan_lock);
        result.plan = plan;
        if (options.prepare_only) {
            ensure_scratch_capacity(*plan);
            result.statistics.prepare_calls = 1;
            result.completion.reset();
            return result;
        }
        if (needs_explicit_fences && !fence_) {
            fence_ = [device_ newFence];
            if (!fence_)
                metal_error("newFence for imported untracked resources");
        }
        for (const auto& c : commands) {
            if (c.kind == Command::Kind::Compute) {
                if (c.compute.control)
                    ++result.statistics.control_dispatches;
                else
                    ++result.statistics.compute_dispatches;
                if (dynamic_dispatch(c.compute))
                    ++result.statistics.control_dispatches;
            } else
                ++result.statistics.blit_commands; // Logical copies/fills, including byte fallback.
        }
        result.statistics.event_waits = options.waits.size();
        result.statistics.event_signals = (options.signal_event ? 1 : 0) + options.signals.size();
        if (commands.empty() && options.waits.empty() && !options.signal_event &&
            options.signals.empty())
            return result;
        completion->buffers.reserve(
            policy == DispatchPolicy::PerKernel ? std::max<size_t>(commands.size(), 1) : 1);
        auto use = std::make_shared<SubmissionUse>(plan);
        use->fence = fence_;
        SubmissionSentinel sentinel{use};
        use->scratch = acquire_scratch(*plan);
        try {
            if (policy == DispatchPolicy::PerKernel && !commands.empty()) {
                // Commit as we go: constructing more uncommitted command buffers than
                // the queue's outstanding-buffer limit could otherwise deadlock.
                for (size_t i = 0; i < commands.size(); ++i) {
                    id<MTLCommandBuffer> cb = new_command_buffer();
                    completion->buffers.push_back({cb});
                    // On a borrowed buffer a later encoding failure cannot
                    // cancel an already encoded prefix. Install ownership
                    // before encoding anything into that external buffer.
                    retain_until_complete(cb, use);
                    if (i == 0)
                        encode_waits(cb, options);
                    encode_direct(cb, *plan->sequence, i, i + 1, use->scratch);
                    if (i + 1 == commands.size())
                        encode_signal(cb, options);
                    if (!externally_managed())
                        [cb commit];
                    ++result.statistics.command_buffers;
                }
            } else {
                id<MTLCommandBuffer> cb = new_command_buffer();
                completion->buffers.push_back({cb});
                retain_until_complete(cb, use);
                encode_waits(cb, options);
                if (policy == DispatchPolicy::Indirect)
                    encode_indirect(cb, *plan, use->scratch);
                else
                    encode_direct(cb, *plan->sequence, 0, commands.size(), use->scratch);
                encode_signal(cb, options);
                if (!externally_managed())
                    [cb commit];
                ++result.statistics.command_buffers;
            }
        } catch (...) {
            // A late host allocation failure may follow earlier per-kernel commits.
            // Drain those commits before propagating the failure, so an unsuccessful
            // submit never leaves work that the Stream cannot synchronize.
            if (!externally_managed())
                for (const auto& cb : completion->buffers)
                    if (cb.buffer.status >= MTLCommandBufferStatusCommitted)
                        [cb.buffer waitUntilCompleted];
            throw;
        }
        return result;
    }
}

id<MTLCommandBuffer> MetalQueue::new_command_buffer() {
    if (command_buffer_) {
        id<MTLCommandBuffer> result = (__bridge id<MTLCommandBuffer>)command_buffer_();
        if (!result)
            throw std::invalid_argument("borrowed Metal command-buffer provider returned null");
        if (result.commandQueue.device.registryID != device_.registryID)
            throw std::invalid_argument("borrowed Metal command buffer belongs to another device");
        if (result.status >= MTLCommandBufferStatusCommitted)
            throw std::invalid_argument("borrowed Metal command buffer has already been committed");
        if (fence_) {
            if (!fence_queue_)
                fence_queue_ = result.commandQueue;
            else if (fence_queue_ != result.commandQueue)
                throw std::invalid_argument("a runtime stream using untracked resources must stay "
                                            "on one native command queue");
        }
        return result;
    }
    id<MTLCommandBuffer> result = [queue_ commandBuffer]; // Retained references enabled.
    if (!result)
        metal_error("commandBuffer");
    result.label = @"metalrt submission";
    return result;
}

void MetalQueue::retain_until_complete(id<MTLCommandBuffer> cb,
                                       const std::shared_ptr<SubmissionUse>& use) {
    auto retained_use = std::make_shared<CommandBufferUse>(use);
    // Borrowed command buffers may use unretained references. Keep the
    // fence in SubmissionUse independently of the stream/native encoder.
    [cb addCompletedHandler:^(id<MTLCommandBuffer> completed) {
      (void)completed;
      retained_use->complete();
    }];
}

void MetalQueue::encode_waits(id<MTLCommandBuffer> cb, const SubmitOptions& options) const {
    for (const auto& point : options.waits) {
        auto event = native_event(point.implementation(), identity_);
        [cb encodeWaitForEvent:event->event value:point.value()];
    }
}

void MetalQueue::encode_signal(id<MTLCommandBuffer> cb, const SubmitOptions& options) const {
    const auto encode = [&](const std::shared_ptr<EventState>& state, uint64_t value) {
        auto event = native_event(state, identity_);
        [cb encodeSignalEvent:event->event value:value];
        // A failed command buffer may not execute its GPU signal. Release
        // downstream GPU waits after failure; retained completions still
        // propagate the producer error instead of reporting success.
        [cb addCompletedHandler:^(id<MTLCommandBuffer> completed) {
          if (completed.status == MTLCommandBufferStatusError) {
              // MTLSharedEvent ignores values below its current value.
              // Do not take the producer's reservation mutex here: a
              // submit can hold it while retiring an older completion,
              // whose wait also waits for this callback to return.
              event->event.signaledValue = value;
          }
        }];
    };
    if (options.signal_event)
        encode(options.signal_event, options.signal_value);
    for (const auto& signal : options.signals)
        encode(signal.event, signal.value);
}

} // namespace metal
} // namespace detail
} // namespace metalrt
