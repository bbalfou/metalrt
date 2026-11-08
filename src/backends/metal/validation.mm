// Native pipeline, resource, and dispatch limit validation.
#include "queue.h"
namespace metalrt {
namespace detail {
namespace metal {
bool MetalQueue::validate(const std::vector<Command>& commands, DispatchPolicy policy) const {
    bool needs_explicit_fences = false;
    const auto validate_resource = [&](const BufferView& view) {
        auto allocation = native_buffer(view, identity_);
        needs_explicit_fences |=
            allocation->buffer.hazardTrackingMode != MTLHazardTrackingModeTracked;
    };
    const MTLSize maximum = device_.maxThreadsPerThreadgroup;
    for (const auto& c : commands) {
        if (c.kind != Command::Kind::Compute) {
            validate_resource(c.destination);
            if (c.kind == Command::Kind::Copy)
                validate_resource(c.source);
            continue;
        }
        const bool icb = policy == DispatchPolicy::Indirect && static_compute(c);
        if (icb && !can_configure_indirect_threadgroup_slots()) {
            for (const auto& argument : c.compute.arguments) {
                if (argument.kind == Argument::Kind::Threadgroup)
                    throw std::runtime_error(
                        "Indirect dynamic threadgroup memory requires macOS 14 or newer and a "
                        "build SDK exposing maxKernelThreadgroupMemoryBindCount; use Direct or "
                        "PerKernel on this configuration");
            }
        }
        auto kernel = std::dynamic_pointer_cast<MetalKernel>(c.compute.kernel.implementation());
        if (!kernel || kernel->device_identity != identity_)
            throw std::invalid_argument("kernel belongs to another device or backend");
        id<MTLComputePipelineState> pipeline = icb ? kernel->indirect : kernel->direct;
        if (!pipeline)
            throw std::runtime_error(
                "Kernel '" + kernel->entry +
                "' does not support indirect commands: " + kernel->indirect_error);
        const auto& group = c.compute.group;
        if (!group.x || !group.y || !group.z ||
            (!c.compute.indirect_groups &&
             (!c.compute.grid.x || !c.compute.grid.y || !c.compute.grid.z)))
            throw std::invalid_argument("Metal dispatch dimensions must be nonzero");
        if (group.x > maximum.width || group.y > maximum.height || group.z > maximum.depth ||
            checked_mul(checked_mul(group.x, group.y), group.z) >
                pipeline.maxTotalThreadsPerThreadgroup)
            throw std::invalid_argument("threadgroup exceeds the Metal device or pipeline limit");
        size_t dynamic_memory = 0;
        const bool derived = c.compute.optional_buffer_mask != UINT32_MAX ||
                             !c.compute.pointer_base_arguments.empty();
        const auto materialized =
            derived ? materialize_compute_arguments(c.compute) : std::vector<Argument>{};
        for (const auto& a : derived ? materialized : c.compute.arguments) {
            if (a.index >= 31)
                throw std::invalid_argument("Metal argument indices must be below 31");
            if (a.kind == Argument::Kind::Buffer)
                validate_resource(a.resource);
            if (a.kind == Argument::Kind::Bytes && a.value.empty())
                throw std::invalid_argument("empty Metal constant argument");
            if (a.kind == Argument::Kind::NullBuffer) {
                const bool reflected = icb ? kernel->indirect_reflected : kernel->direct_reflected;
                const auto active =
                    icb ? kernel->indirect_active_buffers : kernel->direct_active_buffers;
                if (!reflected)
                    throw std::runtime_error("Metal nullable buffer validation requires "
                                             "compute-pipeline argument reflection");
                if (c.compute.optional_buffer_mask == UINT32_MAX &&
                    (active & (uint32_t{1} << a.index)))
                    throw std::invalid_argument(
                        "Kernel '" + kernel->entry + "' has an active null buffer at index " +
                        std::to_string(a.index) +
                        "; configure optional_buffer_mask and a shader that restores null pointers "
                        "from its presence bits");
            }
            if (a.kind == Argument::Kind::Threadgroup) {
                if (a.threadgroup_bytes % 16 != 0)
                    throw std::invalid_argument(
                        "dynamic threadgroup memory lengths must be multiples of 16");
                dynamic_memory = checked_add(dynamic_memory, a.threadgroup_bytes);
            }
        }
        if (checked_add(dynamic_memory, pipeline.staticThreadgroupMemoryLength) >
            device_.maxThreadgroupMemoryLength)
            throw std::invalid_argument("Kernel '" + kernel->entry + "' threadgroup memory (" +
                                        std::to_string(dynamic_memory) + " dynamic + " +
                                        std::to_string(pipeline.staticThreadgroupMemoryLength) +
                                        " static bytes) exceeds the Metal device limit (" +
                                        std::to_string(device_.maxThreadgroupMemoryLength) +
                                        " bytes)");
        const auto validate_control = [&](const BufferView& view, size_t bytes) {
            if (!view)
                return;
            validate_resource(view);
            if (view.offset % 4 != 0 || view.length < bytes)
                throw std::invalid_argument("GPU dispatch control buffers require four-byte "
                                            "alignment and a complete value");
        };
        validate_control(c.compute.indirect_groups, 3 * sizeof(uint32_t));
        validate_control(c.compute.predicate, sizeof(uint32_t));
        validate_control(c.compute.dispatch_status, sizeof(uint32_t));
        if (dynamic_dispatch(c.compute)) {
            const auto groups = c.compute.indirect_groups ? native_size(c.compute.max_groups)
                                                          : threadgroups(c.compute);
            if (!groups.width || !groups.height || !groups.depth || groups.width > UINT32_MAX ||
                groups.height > UINT32_MAX || groups.depth > UINT32_MAX)
                throw std::invalid_argument(
                    "GPU dispatch group bounds must fit nonzero uint32_t values");
        }
    }
    return needs_explicit_fences;
}

} // namespace metal
} // namespace detail
} // namespace metalrt
