// Direct compute, dynamic dispatch, and byte-granular transfers.
#include "queue.h"
namespace metalrt {
namespace detail {
namespace metal {
void MetalQueue::encode_compute(id<MTLComputeCommandEncoder> encoder, const ComputeCommand& command,
                                const PreparedCommand& prepared, id<MTLBuffer> groups,
                                size_t offset) {
    [encoder setComputePipelineState:prepared.kernel->direct];
    for (const auto& arg : prepared.arguments) {
        if (arg.kind == Argument::Kind::Threadgroup)
            [encoder setThreadgroupMemoryLength:arg.threadgroup_bytes atIndex:arg.index];
        else if (arg.kind == Argument::Kind::Texture)
            [encoder setTexture:arg.texture atIndex:arg.index];
        else if (arg.kind == Argument::Kind::Sampler)
            [encoder setSamplerState:arg.sampler atIndex:arg.index];
        else
            [encoder setBuffer:arg.buffer offset:arg.offset atIndex:arg.index];
    }
    if (groups)
        [encoder dispatchThreadgroupsWithIndirectBuffer:groups
                                   indirectBufferOffset:offset
                                  threadsPerThreadgroup:native_size(command.group)];
    else
        [encoder dispatchThreadgroups:threadgroups(command)
                threadsPerThreadgroup:native_size(command.group)];
}

void MetalQueue::encode_transfer(id<MTLCommandBuffer> cb, const Command& command) const {
    auto dst = native_buffer(command.destination, identity_);
    const size_t count = command.destination.length;
    if (count == 0)
        return;
    const bool copy = command.kind == Command::Kind::Copy;
    if (copy && command.source.buffer == command.destination.buffer &&
        command.source.offset == command.destination.offset)
        return;
    auto src = copy ? native_buffer(command.source, identity_) : std::shared_ptr<MetalAllocation>{};
    const bool aligned = command.destination.offset % 4 == 0 && count % 4 == 0 &&
                         (!copy || command.source.offset % 4 == 0);
    if (aligned) {
        id<MTLBlitCommandEncoder> encoder = [cb blitCommandEncoder];
        if (!encoder)
            metal_error("blitCommandEncoder");
        begin_encoder(encoder);
        if (copy)
            [encoder copyFromBuffer:src->buffer
                       sourceOffset:command.source.offset
                           toBuffer:dst->buffer
                  destinationOffset:command.destination.offset
                               size:count];
        else
            [encoder fillBuffer:dst->buffer
                          range:NSMakeRange(command.destination.offset, count)
                          value:command.fill_value];
        end_encoder(encoder);
        return;
    }
    id<MTLComputeCommandEncoder> encoder =
        [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeSerial];
    if (!encoder)
        metal_error("byte transfer compute encoder");
    begin_encoder(encoder);
    constexpr size_t group = 64;
    size_t grid = std::min<size_t>(count, 65536);
    size_t groups = 1 + (grid - 1) / group;
    uint64_t stride = groups * group;
    if (copy) {
        const uint64_t args[4] = {command.source.offset, command.destination.offset, count, stride};
        [encoder setComputePipelineState:transfers_->copy];
        [encoder setBuffer:src->buffer offset:0 atIndex:0];
        [encoder setBuffer:dst->buffer offset:0 atIndex:1];
        [encoder setBytes:args length:sizeof(args) atIndex:2];
    } else {
        const uint64_t args[4] = {command.destination.offset, count, stride, command.fill_value};
        [encoder setComputePipelineState:transfers_->fill];
        [encoder setBuffer:dst->buffer offset:0 atIndex:0];
        [encoder setBytes:args length:sizeof(args) atIndex:1];
    }
    [encoder dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(group, 1, 1)];
    end_encoder(encoder);
}

void MetalQueue::encode_dynamic(id<MTLCommandBuffer> cb, const PreparedSequence& sequence,
                                size_t index,
                                const std::shared_ptr<DispatchScratch>& scratch) const {
    const auto& command = sequence.commands[index].compute;
    const auto& prepared = sequence.compute[index];
    const size_t offset = sequence.scratch_offsets[index];
    if (!scratch || offset == SIZE_MAX)
        throw std::logic_error("missing GPU dispatch scratch lease");
    struct DispatchConfig {
        uint32_t fixed[3], maximum[3], flags;
    } config{};
    static_assert(sizeof(DispatchConfig) == 7 * sizeof(uint32_t), "MSL dispatch configuration ABI");
    const auto groups = command.indirect_groups ? MTLSizeMake(0, 0, 0) : threadgroups(command);
    const auto maximum = command.indirect_groups ? native_size(command.max_groups) : groups;
    config.fixed[0] = uint32_t(groups.width);
    config.fixed[1] = uint32_t(groups.height);
    config.fixed[2] = uint32_t(groups.depth);
    config.maximum[0] = uint32_t(maximum.width);
    config.maximum[1] = uint32_t(maximum.height);
    config.maximum[2] = uint32_t(maximum.depth);
    config.flags = (command.indirect_groups ? 1u : 0u) | (command.predicate ? 2u : 0u) |
                   (command.dispatch_status ? 4u : 0u);

    id<MTLComputeCommandEncoder> prepare =
        [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeSerial];
    if (!prepare)
        metal_error("dispatch preparation encoder");
    begin_encoder(prepare);
    [prepare setComputePipelineState:transfers_->prepare_dispatch];
    const auto bind_control = [&](const BufferView& view, NSUInteger slot) {
        if (view)
            [prepare setBuffer:native_buffer(view, identity_)->buffer
                        offset:view.offset
                       atIndex:slot];
        // All declared MSL buffer arguments have a valid binding. A missing
        // optional input aliases scratch but is never accessed by the shader.
        else
            [prepare setBuffer:scratch->buffer offset:offset atIndex:slot];
    };
    bind_control(command.indirect_groups, 0);
    bind_control(command.predicate, 1);
    [prepare setBuffer:scratch->buffer offset:offset atIndex:2];
    bind_control(command.dispatch_status, 3);
    [prepare setBytes:&config length:sizeof(config) atIndex:4];
    [prepare dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    end_encoder(prepare);

    // Dispatch arguments must be available before the consuming compute
    // pass. This boundary orders the GPU preparation kernel and the payload,
    // including on older Metal devices; there is no CPU readback or wait.
    id<MTLComputeCommandEncoder> payload =
        [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeSerial];
    if (!payload)
        metal_error("GPU-sized dispatch encoder");
    begin_encoder(payload);
    encode_compute(payload, command, prepared, scratch->buffer, offset);
    end_encoder(payload);
}

void MetalQueue::encode_direct(id<MTLCommandBuffer> cb, const PreparedSequence& sequence,
                               size_t begin, size_t end,
                               const std::shared_ptr<DispatchScratch>& scratch) const {
    size_t i = begin;
    while (i < end) {
        if (sequence.commands[i].kind != Command::Kind::Compute) {
            encode_transfer(cb, sequence.commands[i]);
            ++i;
            continue;
        }
        if (dynamic_dispatch(sequence.commands[i].compute)) {
            encode_dynamic(cb, sequence, i, scratch);
            ++i;
            continue;
        }
        // Explicit barriers preserve the conservative default while commands
        // with barrier_before=false may overlap inside a parallel group.
        id<MTLComputeCommandEncoder> encoder =
            [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
        if (!encoder)
            metal_error("computeCommandEncoderWithDispatchType");
        begin_encoder(encoder);
        const size_t first = i;
        do {
            if (i > first && sequence.commands[i].compute.barrier_before)
                [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers | MTLBarrierScopeTextures];
            encode_compute(encoder, sequence.commands[i].compute, sequence.compute[i]);
            ++i;
        } while (i < end && static_compute(sequence.commands[i]));
        end_encoder(encoder);
    }
}

} // namespace metal
} // namespace detail
} // namespace metalrt
