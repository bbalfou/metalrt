#pragma once
// Native resources, immutable prepared commands, and in-flight ownership.
// Handwritten Objective-C++ backend. Requires ARC and macOS 13 or newer.
#include <metalrt/detail/backend.h>
#include <metalrt/detail/cache_key.h>
#include <metalrt/interop.h>
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <algorithm>
#include <array>
#include <filesystem>
#include <limits>
#include <set>
#include <sstream>
#include <utility>

namespace metalrt {
namespace detail {
namespace metal {
inline std::string ns_string(NSString* value) {
    const char* text = value ? value.UTF8String : nullptr;
    return text ? text : "unknown Metal error";
}
[[noreturn]] inline void metal_error(const std::string& operation, NSError* error = nullptr) {
    throw std::runtime_error(operation +
                             (error ? ": " + ns_string(error.localizedDescription) : " failed"));
}
inline MTLSize native_size(Size3 s) {
    return MTLSizeMake(s.x, s.y, s.z);
}
inline bool can_configure_indirect_threadgroup_slots() {
#if __MAC_OS_X_VERSION_MAX_ALLOWED >= 140000
    if (@available(macOS 14.0, *))
        return true;
#endif
    return false;
}
inline MTLSize threadgroups(const ComputeCommand& c) {
    return MTLSizeMake(1 + (c.grid.x - 1) / c.group.x, 1 + (c.grid.y - 1) / c.group.y,
                       1 + (c.grid.z - 1) / c.group.z);
}
inline MTLResourceUsage resource_usage(Access access) {
    switch (access) {
    case Access::Read:
        return MTLResourceUsageRead;
    case Access::Write:
        return MTLResourceUsageWrite;
    case Access::ReadWrite:
        return MTLResourceUsageRead | MTLResourceUsageWrite;
    }
    throw std::invalid_argument("invalid resource access mode");
}

struct MetalAllocation final : BufferAllocation {
    id<MTLBuffer> buffer = nil;
    // Framework caching allocators can reuse a live native object after their
    // storage handle dies. Keep allocator leases, not only the Objective-C
    // object. Mutations are serialized by the owning MetalDevice's mutex.
    std::unordered_map<const void*, std::shared_ptr<void>> external_owners;
    void* contents() const override {
        return storage == StorageMode::Shared ? buffer.contents : nullptr;
    }
};
struct MetalKernel final : KernelState {
    id<MTLFunction> function = nil;
    id<MTLComputePipelineState> direct = nil;
    id<MTLComputePipelineState> indirect = nil;
    std::string indirect_error;
    std::set<std::string> archive_locations;
    std::array<size_t, 31> minimum_buffer_bytes{};
    uint32_t direct_active_buffers = 0, indirect_active_buffers = 0;
    bool direct_reflected = false, indirect_reflected = false;
};
inline bool reflect_buffers(MTLComputePipelineReflection* reflection, uint32_t& active,
                            std::array<size_t, 31>& minimum_bytes) {
    if (!reflection)
        return false;
    for (MTLArgument* argument in reflection.arguments) {
        if (argument.type != MTLArgumentTypeBuffer || argument.index >= minimum_bytes.size())
            continue;
        const auto index = static_cast<uint32_t>(argument.index);
        if (argument.active)
            active |= uint32_t{1} << index;
        // For a struct this is its full byte size; for arrays/pointers it is
        // one element. ArgumentInfo provides this without detailed type trees.
        minimum_bytes[index] = std::max(minimum_bytes[index], size_t(argument.bufferDataSize));
    }
    return true;
}
struct MetalEvent final : EventState {
    id<MTLSharedEvent> event = nil;
    uint64_t completed_value() const override { return event.signaledValue; }
};
struct LibraryHolder {
    id<MTLLibrary> library = nil;
};
struct BufferHolder {
    id<MTLBuffer> buffer = nil;
};
struct CommandBufferHolder {
    id<MTLCommandBuffer> buffer = nil;
};
struct NativeArchive {
    id<MTLBinaryArchive> object = nil;
    std::filesystem::path path;
    std::string directory;
    bool requested = false, loaded = false, dirty = false;
};

inline std::shared_ptr<MetalAllocation> native_buffer(const BufferView& view,
                                                      const void* identity) {
    auto result = std::dynamic_pointer_cast<MetalAllocation>(view.buffer.implementation());
    if (!result || result->device_identity != identity)
        throw std::invalid_argument("buffer belongs to another device or backend");
    if (view.offset > result->bytes || view.length > result->bytes - view.offset)
        throw std::out_of_range("native buffer view exceeds its allocation");
    return result;
}
inline std::shared_ptr<MetalEvent> native_event(const std::shared_ptr<EventState>& event,
                                                const void* identity) {
    auto result = std::dynamic_pointer_cast<MetalEvent>(event);
    if (!result || result->device_identity != identity)
        throw std::invalid_argument("event belongs to another device or backend");
    return result;
}

inline bool dynamic_dispatch(const ComputeCommand& command) {
    return command.indirect_groups || command.predicate || command.dispatch_status;
}
inline bool static_compute(const Command& command) {
    return command.kind == Command::Kind::Compute && !dynamic_dispatch(command.compute);
}
inline bool same_size(Size3 a, Size3 b) {
    return a.x == b.x && a.y == b.y && a.z == b.z;
}
// This allocation-free comparison allows an edited graph to retain the existing
// immutable constant buffers of all commands that the edit did not change.
inline bool same_compute(const ComputeCommand& a, const ComputeCommand& b) {
    if (!(a.kernel == b.kernel) || !same_size(a.grid, b.grid) || !same_size(a.group, b.group) ||
        a.barrier_before != b.barrier_before || a.control != b.control ||
        !(a.indirect_groups == b.indirect_groups) || !same_size(a.max_groups, b.max_groups) ||
        !(a.predicate == b.predicate) || !(a.dispatch_status == b.dispatch_status) ||
        a.arguments.size() != b.arguments.size() ||
        a.optional_buffer_mask != b.optional_buffer_mask ||
        a.pointer_base_arguments != b.pointer_base_arguments)
        return false;
    for (size_t i = 0; i < a.arguments.size(); ++i) {
        const auto& x = a.arguments[i];
        const auto& y = b.arguments[i];
        if (x.index != y.index || x.kind != y.kind || x.access != y.access ||
            !(x.resource == y.resource) || x.value != y.value ||
            x.threadgroup_bytes != y.threadgroup_bytes)
            return false;
    }
    return true;
}

struct MetalCompletion final : Completion {
    std::vector<CommandBufferHolder> buffers;
    std::vector<EventPoint> dependencies;
    std::shared_ptr<TimingState> timing;
    std::once_flag timing_collected;
    bool externally_managed = false;
    bool append_dependencies(std::vector<Ticket>& output, bool) override {
        for (const auto& point : dependencies)
            output.push_back(point.completion());
        return true;
    }
    void wait() override {
        @autoreleasepool {
            std::string first_error;
            // Ticket observes the retained event dependencies (and their errors)
            // iteratively before observing these native command buffers.
            for (const auto& holder : buffers) {
                if (externally_managed && holder.buffer.status < MTLCommandBufferStatusCommitted)
                    throw UncommittedCommandBufferError(
                        "cannot wait for an uncommitted borrowed Metal command buffer; its "
                        "execution owner must commit it first");
                [holder.buffer waitUntilCompleted];
                if (holder.buffer.status == MTLCommandBufferStatusError && first_error.empty())
                    first_error = ns_string(holder.buffer.error.localizedDescription);
            }
            // Apple defines these timestamps only after command completion.
            // Retiring a Ticket, waiting on an event, and synchronizing may all
            // observe this completion: count each buffer exactly once.
            if (timing)
                std::call_once(timing_collected, [this] {
                    for (const auto& holder : buffers)
                        timing->record(holder.buffer.GPUStartTime, holder.buffer.GPUEndTime,
                                       holder.buffer.status == MTLCommandBufferStatusCompleted);
                });
            if (!first_error.empty())
                throw std::runtime_error("Metal command buffer failed: " + first_error);
        }
    }
    bool ready() const override {
        for (const auto& holder : buffers) {
            auto status = holder.buffer.status;
            if (status != MTLCommandBufferStatusCompleted && status != MTLCommandBufferStatusError)
                return false;
        }
        return true;
    }
};

struct PreparedArgument {
    uint32_t index = 0;
    Argument::Kind kind = Argument::Kind::Buffer;
    id<MTLBuffer> buffer = nil;
    size_t offset = 0, threadgroup_bytes = 0;
    Access access = Access::Read;
};
struct PreparedCommand {
    std::shared_ptr<MetalKernel> kernel;
    std::vector<PreparedArgument> arguments;
};
struct PreparedSequence {
    std::vector<Command> commands; // Keeps views and all source kernel objects alive.
    std::vector<PreparedCommand> compute;
    std::vector<size_t> scratch_offsets;
    size_t scratch_bytes = 0;
};
struct IndirectStorage {
    id<MTLIndirectCommandBuffer> icb = nil;
    size_t buffer_slots = 0, threadgroup_slots = 0;
    std::vector<ComputeCommand> commands;
    std::vector<PreparedCommand> compute;
    std::vector<std::pair<BufferHolder, MTLResourceUsage>> resources;
};
struct IndirectSegment {
    size_t begin = 0, count = 0, buffer_slots = 0, threadgroup_slots = 0;
    std::shared_ptr<IndirectStorage> storage;
};
struct DispatchScratch {
    id<MTLBuffer> buffer = nil;
    size_t bytes = 0;
    std::atomic<bool> busy{false};
};
struct MetalPlan final : Plan {
    const void* device_identity = nullptr;
    DispatchPolicy policy = DispatchPolicy::Direct;
    std::mutex mutex;
    std::atomic<size_t> in_flight{0};
    std::shared_ptr<PreparedSequence> sequence;
    std::vector<IndirectSegment> segments;
    std::vector<std::shared_ptr<DispatchScratch>> scratch;
};
// All command buffers of one submission share this lease. The sentinel accounts
// for the host still encoding later PerKernel command buffers. Scratch is released
// by the last completion, not by the lifetime of the caller's Ticket object.
struct SubmissionUse {
    std::shared_ptr<MetalPlan> plan;
    std::shared_ptr<DispatchScratch> scratch;
    id<MTLFence> fence = nil;
    std::atomic<size_t> pending{1};
    explicit SubmissionUse(std::shared_ptr<MetalPlan> p) : plan(std::move(p)) {
        plan->in_flight.fetch_add(1, std::memory_order_release);
    }
    void add_command_buffer() { pending.fetch_add(1, std::memory_order_relaxed); }
    void complete() {
        if (pending.fetch_sub(1, std::memory_order_acq_rel) == 1) {
            if (scratch)
                scratch->busy.store(false, std::memory_order_release);
            plan->in_flight.fetch_sub(1, std::memory_order_release);
        }
    }
};
struct SubmissionSentinel {
    std::shared_ptr<SubmissionUse> use;
    ~SubmissionSentinel() { use->complete(); }
};
// A caller may cancel a borrowed command buffer by destroying it without ever
// committing. Retire its plan/scratch lease in that case as well as when its
// completion handler runs. The completion and destruction paths are idempotent.
struct CommandBufferUse {
    std::shared_ptr<SubmissionUse> use;
    std::atomic<bool> completed{false};
    explicit CommandBufferUse(std::shared_ptr<SubmissionUse> value) : use(std::move(value)) {
        use->add_command_buffer();
    }
    ~CommandBufferUse() { complete(); }
    void complete() {
        if (!completed.exchange(true, std::memory_order_acq_rel))
            use->complete();
    }
};

// macOS blits require four-byte-aligned ranges. The byte kernels preserve the
// byte-granular memory API for private/shared buffers; their grid-stride loops
// also cover allocations larger than 2^32 bytes. The dispatch preparation kernel
// bounds GPU-produced arguments and predicates without a host round trip.
constexpr const char* byte_transfer_source = R"METAL(
#include <metal_stdlib>
using namespace metal;
struct CopyBytes { ulong source; ulong destination; ulong count; ulong stride; };
struct FillBytes { ulong destination; ulong count; ulong stride; ulong value; };
kernel void metalrt_copy_bytes(device const uchar* src [[buffer(0)]],
                               device uchar* dst [[buffer(1)]],
                               constant CopyBytes& a [[buffer(2)]],
                               uint tid [[thread_position_in_grid]]) {
    for(ulong i=tid; i<a.count; i+=a.stride) dst[a.destination+i]=src[a.source+i];
}
kernel void metalrt_fill_bytes(device uchar* dst [[buffer(0)]],
                               constant FillBytes& a [[buffer(1)]],
                               uint tid [[thread_position_in_grid]]) {
    for(ulong i=tid; i<a.count; i+=a.stride) dst[a.destination+i]=uchar(a.value);
}
// The fixed topology can use GPU-produced grid sizes and predicates without
// reading either on the CPU. Inactive payloads receive zero threadgroups.
struct DispatchConfig {
    uint fixed_groups[3];
    uint maximum_groups[3];
    uint flags; // bit 0: dynamic groups, bit 1: predicate, bit 2: status output
};
kernel void metalrt_prepare_dispatch(device const uint* requested [[buffer(0)]],
                                    device const uint* predicate [[buffer(1)]],
                                    device uint* output [[buffer(2)]],
                                    device uint* status [[buffer(3)]],
                                    constant DispatchConfig& config [[buffer(4)]]) {
    uint3 groups=uint3(config.fixed_groups[0],config.fixed_groups[1],config.fixed_groups[2]);
    if(config.flags&1u) groups=uint3(requested[0],requested[1],requested[2]);
    const uint3 maximum=uint3(config.maximum_groups[0],config.maximum_groups[1],config.maximum_groups[2]);
    const bool clamped=any(groups>maximum);
    groups=min(groups,maximum);
    const bool active=!(config.flags&2u) || predicate[0]!=0;
    if(!active) groups=uint3(0);
    output[0]=groups.x; output[1]=groups.y; output[2]=groups.z;
    if(config.flags&4u) status[0]=(clamped ? 1u : 0u) | (active ? 0u : 2u);
}
)METAL";
struct TransferPipelines {
    id<MTLComputePipelineState> copy = nil, fill = nil, prepare_dispatch = nil;
};

} // namespace metal
} // namespace detail
} // namespace metalrt
