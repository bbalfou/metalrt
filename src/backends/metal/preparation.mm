// Immutable bindings, reusable plans, and per-submission dispatch scratch.
#include "queue.h"
namespace metalrt {
namespace detail {
namespace metal {
void MetalQueue::record_allocation(size_t bytes) const {
    metrics_->buffer_allocations.fetch_add(1, std::memory_order_relaxed);
    metrics_->allocated_bytes.fetch_add(bytes, std::memory_order_relaxed);
}

std::shared_ptr<PreparedSequence>
MetalQueue::prepare(const std::vector<Command>& commands,
                    const std::shared_ptr<PreparedSequence>& previous) const {
    auto sequence = std::make_shared<PreparedSequence>();
    sequence->commands = commands;
    sequence->compute.resize(commands.size());
    sequence->scratch_offsets.resize(commands.size(), SIZE_MAX);
    for (size_t i = 0; i < commands.size(); ++i) {
        if (commands[i].kind != Command::Kind::Compute)
            continue;
        const auto& c = commands[i].compute;
        auto& out = sequence->compute[i];
        if (dynamic_dispatch(c)) {
            sequence->scratch_offsets[i] = sequence->scratch_bytes;
            // A private 16-byte lane holds the three uint32_t group counts.
            sequence->scratch_bytes = checked_add(sequence->scratch_bytes, 16);
        }
        if (previous && i < previous->commands.size() &&
            previous->commands[i].kind == Command::Kind::Compute &&
            same_compute(c, previous->commands[i].compute)) {
            out = previous->compute[i];
            continue;
        }
        out.kernel = std::static_pointer_cast<MetalKernel>(c.kernel.implementation());
        const bool derived =
            c.optional_buffer_mask != UINT32_MAX || !c.pointer_base_arguments.empty();
        const auto materialized =
            derived ? materialize_compute_arguments(c) : std::vector<Argument>{};
        const auto& arguments = derived ? materialized : c.arguments;
        out.arguments.reserve(arguments.size());
        for (const auto& a : arguments) {
            PreparedArgument p;
            p.index = a.index;
            p.kind = a.kind;
            p.access = a.access;
            p.threadgroup_bytes = a.threadgroup_bytes;
            if (a.kind == Argument::Kind::Buffer) {
                p.buffer = native_buffer(a.resource, identity_)->buffer;
                p.offset = a.resource.offset;
            } else if (a.kind == Argument::Kind::Texture) {
                auto texture = std::dynamic_pointer_cast<MetalTexture>(a.image.implementation());
                if (!texture || texture->device_identity != identity_)
                    throw std::invalid_argument("texture belongs to another backend or device");
                p.texture = texture->texture;
            } else if (a.kind == Argument::Kind::Sampler) {
                auto sampler = std::dynamic_pointer_cast<MetalSampler>(a.sampling.implementation());
                if (!sampler || sampler->device_identity != identity_)
                    throw std::invalid_argument("sampler belongs to another backend or device");
                p.sampler = sampler->sampler;
            } else if (a.kind == Argument::Kind::Bytes) {
                p.buffer = [device_ newBufferWithBytes:a.value.data()
                                                length:a.value.size()
                                               options:MTLResourceStorageModeShared |
                                                       MTLResourceHazardTrackingModeTracked];
                if (!p.buffer)
                    metal_error("allocating immutable kernel constants");
                record_allocation(a.value.size());
                p.access = Access::Read;
            } else if (a.kind == Argument::Kind::NullBuffer &&
                       c.optional_buffer_mask != UINT32_MAX) {
                // API validation requires an actual resource for active
                // buffer parameters even when the shader skips all uses.
                // The independent presence mask preserves semantic null;
                // these bytes are never a substitute value for user data.
                const auto minimum =
                    std::max<size_t>(16, out.kernel->minimum_buffer_bytes[a.index]);
                const auto bytes = checked_add(minimum, size_t{15}) & ~size_t{15};
                if (bytes > device_.maxBufferLength)
                    throw std::length_error(
                        "optional kernel buffer fallback exceeds MTLDevice.maxBufferLength");
                p.buffer = [device_ newBufferWithLength:bytes
                                                options:MTLResourceStorageModeShared |
                                                        MTLResourceHazardTrackingModeTracked];
                if (!p.buffer)
                    metal_error("allocating optional kernel buffer fallback");
                std::memset(p.buffer.contents, 0, bytes);
                record_allocation(bytes);
                p.kind = Argument::Kind::Buffer;
            }
            out.arguments.push_back(std::move(p));
        }
    }
    return sequence;
}

std::shared_ptr<DispatchScratch> MetalQueue::make_scratch(size_t bytes) const {
    if (bytes > device_.maxBufferLength)
        throw std::length_error("GPU dispatch scratch exceeds MTLDevice.maxBufferLength");
    auto result = std::make_shared<DispatchScratch>();
    result->bytes = bytes;
    result->buffer = [device_
        newBufferWithLength:bytes
                    options:MTLResourceStorageModePrivate | MTLResourceHazardTrackingModeTracked];
    if (!result->buffer)
        metal_error("allocating GPU dispatch scratch");
    record_allocation(bytes);
    return result;
}

void MetalQueue::ensure_scratch_capacity(MetalPlan& plan) const {
    const auto bytes = plan.sequence->scratch_bytes;
    if (!bytes)
        return;
    for (const auto& scratch : plan.scratch)
        if (scratch->bytes >= bytes && !scratch->busy.load(std::memory_order_acquire))
            return;
    plan.scratch.push_back(make_scratch(bytes));
}

std::shared_ptr<DispatchScratch> MetalQueue::acquire_scratch(MetalPlan& plan) const {
    const auto bytes = plan.sequence->scratch_bytes;
    if (!bytes)
        return {};
    for (const auto& scratch : plan.scratch) {
        bool expected = false;
        if (scratch->bytes >= bytes &&
            scratch->busy.compare_exchange_strong(expected, true, std::memory_order_acq_rel))
            return scratch;
    }
    auto scratch = make_scratch(bytes);
    plan.scratch.push_back(scratch); // Finish potentially throwing work before leasing it.
    scratch->busy.store(true, std::memory_order_release);
    return scratch;
}

std::shared_ptr<MetalPlan>
MetalQueue::prepare_plan(const std::vector<Command>& commands, DispatchPolicy policy,
                         std::shared_ptr<Plan> cached, Statistics& statistics,
                         std::unique_lock<std::mutex>& result_lock) const {
    auto previous = std::dynamic_pointer_cast<MetalPlan>(cached);
    std::unique_lock<std::mutex> previous_lock;
    if (previous && previous->device_identity == identity_ && previous->policy == policy)
        previous_lock = std::unique_lock<std::mutex>(previous->mutex);
    else
        previous.reset();
    if (previous && same_commands(previous->sequence->commands, commands)) {
        statistics.icb_reuses = previous->segments.size();
        result_lock = std::move(previous_lock);
        return previous;
    }

    auto sequence = prepare(commands, previous ? previous->sequence : nullptr);
    auto layout = policy == DispatchPolicy::Indirect ? segment_layout(*sequence)
                                                     : std::vector<IndirectSegment>{};
    const bool idle = previous && previous->in_flight.load(std::memory_order_acquire) == 0;
    auto plan = idle ? previous : std::make_shared<MetalPlan>();
    if (idle)
        result_lock = std::move(previous_lock);
    else
        result_lock = std::unique_lock<std::mutex>(plan->mutex);
    auto scratch = previous ? previous->scratch : std::vector<std::shared_ptr<DispatchScratch>>{};
    if (sequence->scratch_bytes &&
        std::none_of(scratch.begin(), scratch.end(),
                     [&](const auto& item) { return item->bytes >= sequence->scratch_bytes; }))
        scratch.push_back(make_scratch(sequence->scratch_bytes));
    std::vector<std::shared_ptr<IndirectStorage>> to_encode;
    to_encode.reserve(layout.size());
    for (auto& segment : layout) {
        const IndirectSegment* old = nullptr;
        if (previous)
            for (const auto& candidate : previous->segments) {
                if (candidate.begin == segment.begin && candidate.count == segment.count) {
                    old = &candidate;
                    break;
                }
            }
        if (old && same_segment(*old->storage, *sequence, segment)) {
            segment.storage = old->storage;
            ++statistics.icb_reuses;
            continue;
        }
        segment.storage = make_storage(*sequence, segment);
        // A static ICB may be shared by multiple immutable plan versions when
        // only dynamic nodes change. Such shared storage is never rewritten.
        if (idle && old && old->storage.use_count() == 1 &&
            old->storage->buffer_slots >= segment.buffer_slots &&
            old->storage->threadgroup_slots >= segment.threadgroup_slots) {
            segment.storage->icb = old->storage->icb;
            segment.storage->buffer_slots = old->storage->buffer_slots;
            segment.storage->threadgroup_slots = old->storage->threadgroup_slots;
            ++statistics.icb_updates;
        } else {
            allocate_icb(*segment.storage);
            ++statistics.icb_builds;
        }
        to_encode.push_back(segment.storage);
    }
    // All host allocations have succeeded before an idle old ICB is reset.
    // Unchanged segments retain their exact pipeline/constant/buffer bindings.
    for (const auto& storage : to_encode)
        encode_icb(*storage);
    plan->device_identity = identity_;
    plan->policy = policy;
    plan->sequence = std::move(sequence);
    plan->segments = std::move(layout);
    plan->scratch = std::move(scratch);
    return plan;
}

} // namespace metal
} // namespace detail
} // namespace metalrt
