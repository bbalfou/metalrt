// Static compute segments encoded into reusable indirect command buffers.
#include "queue.h"
namespace metalrt {
namespace detail {
namespace metal {
std::vector<IndirectSegment> MetalQueue::segment_layout(const PreparedSequence& sequence) {
    std::vector<IndirectSegment> result;
    size_t i = 0;
    while (i < sequence.commands.size()) {
        if (!static_compute(sequence.commands[i])) {
            ++i;
            continue;
        }
        IndirectSegment segment;
        segment.begin = i;
        // 16384 also respects the Mac-family maximum execution range.
        while (i < sequence.commands.size() && static_compute(sequence.commands[i]) &&
               segment.count < 16384) {
            for (const auto& a : sequence.compute[i].arguments) {
                if (a.kind == Argument::Kind::Threadgroup)
                    segment.threadgroup_slots =
                        std::max(segment.threadgroup_slots, size_t(a.index) + 1);
                else
                    segment.buffer_slots = std::max(segment.buffer_slots, size_t(a.index) + 1);
            }
            ++i;
            ++segment.count;
        }
        result.push_back(std::move(segment));
    }
    return result;
}

bool MetalQueue::same_segment(const IndirectStorage& previous, const PreparedSequence& next,
                              const IndirectSegment& segment) {
    if (previous.commands.size() != segment.count)
        return false;
    for (size_t j = 0; j < segment.count; ++j)
        if (!same_compute(previous.commands[j], next.commands[segment.begin + j].compute))
            return false;
    return true;
}

void MetalQueue::allocate_icb(IndirectStorage& storage) const {
    if (storage.threadgroup_slots && !can_configure_indirect_threadgroup_slots())
        throw std::runtime_error("Indirect dynamic threadgroup memory requires macOS 14 or newer "
                                 "and a build SDK exposing maxKernelThreadgroupMemoryBindCount; "
                                 "use Direct or PerKernel on this configuration");
    MTLIndirectCommandBufferDescriptor* descriptor = [MTLIndirectCommandBufferDescriptor new];
    descriptor.commandTypes = MTLIndirectCommandTypeConcurrentDispatch;
    descriptor.inheritPipelineState = NO;
    descriptor.inheritBuffers = NO;
    descriptor.maxKernelBufferBindCount = storage.buffer_slots;
#if __MAC_OS_X_VERSION_MAX_ALLOWED >= 140000
    if (@available(macOS 14.0, *))
        descriptor.maxKernelThreadgroupMemoryBindCount = storage.threadgroup_slots;
#endif
    storage.icb =
        [device_ newIndirectCommandBufferWithDescriptor:descriptor
                                        maxCommandCount:storage.commands.size()
                                                options:MTLResourceStorageModeShared |
                                                        MTLResourceHazardTrackingModeTracked];
    if (!storage.icb)
        metal_error("newIndirectCommandBufferWithDescriptor");
}

void MetalQueue::encode_icb(const IndirectStorage& storage) {
    for (size_t j = 0; j < storage.commands.size(); ++j) {
        const auto& command = storage.commands[j];
        const auto& prepared = storage.compute[j];
        id<MTLIndirectComputeCommand> indirect = [storage.icb indirectComputeCommandAtIndex:j];
        [indirect reset];
        // ICB compute commands are concurrent even in a serial parent encoder.
        // A barrier precedes each dependent dispatch inside the range.
        if (j > 0 && command.barrier_before)
            [indirect setBarrier];
        else
            [indirect clearBarrier];
        [indirect setComputePipelineState:prepared.kernel->indirect];
        for (const auto& a : prepared.arguments) {
            if (a.kind == Argument::Kind::Threadgroup)
                [indirect setThreadgroupMemoryLength:a.threadgroup_bytes atIndex:a.index];
            // reset() clears the buffer table. Unlike the ordinary
            // encoder API, setKernelBuffer takes a non-null buffer.
            else if (a.kind != Argument::Kind::NullBuffer)
                [indirect setKernelBuffer:a.buffer offset:a.offset atIndex:a.index];
        }
        [indirect concurrentDispatchThreadgroups:threadgroups(command)
                           threadsPerThreadgroup:native_size(command.group)];
    }
}

std::shared_ptr<IndirectStorage> MetalQueue::make_storage(const PreparedSequence& sequence,
                                                          const IndirectSegment& segment) {
    auto storage = std::make_shared<IndirectStorage>();
    storage->buffer_slots = segment.buffer_slots;
    storage->threadgroup_slots = segment.threadgroup_slots;
    storage->commands.reserve(segment.count);
    storage->compute.reserve(segment.count);
    std::unordered_map<void*, std::pair<BufferHolder, MTLResourceUsage>> resources;
    for (size_t i = segment.begin; i < segment.begin + segment.count; ++i) {
        storage->commands.push_back(sequence.commands[i].compute);
        storage->compute.push_back(sequence.compute[i]);
        for (const auto& arg : sequence.compute[i].arguments) {
            if (arg.kind == Argument::Kind::Threadgroup || !arg.buffer)
                continue;
            auto& resource = resources[(__bridge void*)arg.buffer];
            resource.first.buffer = arg.buffer;
            resource.second = resource.second | resource_usage(arg.access);
        }
    }
    storage->resources.reserve(resources.size());
    for (auto& resource : resources)
        storage->resources.push_back(std::move(resource.second));
    return storage;
}

void MetalQueue::encode_indirect(id<MTLCommandBuffer> cb, const MetalPlan& plan,
                                 const std::shared_ptr<DispatchScratch>& scratch) const {
    const auto& sequence = *plan.sequence;
    const auto& commands = sequence.commands;
    size_t cursor = 0, segment_index = 0;
    while (cursor < commands.size()) {
        if (commands[cursor].kind != Command::Kind::Compute) {
            encode_transfer(cb, commands[cursor]);
            ++cursor;
            continue;
        }
        if (dynamic_dispatch(commands[cursor].compute)) {
            encode_dynamic(cb, sequence, cursor, scratch);
            ++cursor;
            continue;
        }
        const auto& segment = plan.segments.at(segment_index++);
        id<MTLComputeCommandEncoder> encoder =
            [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeSerial];
        if (!encoder)
            metal_error("ICB parent compute encoder");
        begin_encoder(encoder);
        // Usage is the union across the entire static pass, calculated once
        // during preparation, including the immutable scalar buffers.
        for (const auto& resource : segment.storage->resources)
            [encoder useResource:resource.first.buffer usage:resource.second];
        [encoder executeCommandsInBuffer:segment.storage->icb
                               withRange:NSMakeRange(0, segment.count)];
        end_encoder(encoder);
        cursor += segment.count;
    }
}

} // namespace metal
} // namespace detail
} // namespace metalrt
