// Backend-independent validation and immutable command comparison.
#include "runtime_internal.h"

namespace metalrt {
namespace detail {
bool same_commands(const std::vector<Command>& a, const std::vector<Command>& b) {
    if (a.size() != b.size())
        return false;
    for (size_t i = 0; i < a.size(); ++i) {
        const auto& x = a[i];
        const auto& y = b[i];
        if (x.kind != y.kind)
            return false;
        if (x.kind == Command::Kind::Copy) {
            if (!(x.source == y.source) || !(x.destination == y.destination))
                return false;
        } else if (x.kind == Command::Kind::Fill) {
            if (!(x.destination == y.destination) || x.fill_value != y.fill_value)
                return false;
        } else {
            const auto& p = x.compute;
            const auto& q = y.compute;
            if (!(p.indirect_groups == q.indirect_groups) || !(p.predicate == q.predicate) ||
                !(p.dispatch_status == q.dispatch_status) || p.max_groups.x != q.max_groups.x ||
                p.max_groups.y != q.max_groups.y || p.max_groups.z != q.max_groups.z ||
                p.control != q.control || p.optional_buffer_mask != q.optional_buffer_mask ||
                p.pointer_base_arguments != q.pointer_base_arguments)
                return false;
            if (!(p.kernel == q.kernel) || p.barrier_before != q.barrier_before ||
                p.grid.x != q.grid.x || p.grid.y != q.grid.y || p.grid.z != q.grid.z ||
                p.group.x != q.group.x || p.group.y != q.group.y || p.group.z != q.group.z ||
                p.arguments.size() != q.arguments.size())
                return false;
            for (size_t j = 0; j < p.arguments.size(); ++j) {
                const auto& c = p.arguments[j];
                const auto& d = q.arguments[j];
                if (c.kind != d.kind || c.index != d.index || c.access != d.access ||
                    !(c.resource == d.resource) || !(c.image == d.image) ||
                    !(c.sampling == d.sampling) || c.value != d.value ||
                    c.threadgroup_bytes != d.threadgroup_bytes)
                    return false;
            }
        }
    }
    return true;
}
} // namespace detail
bool overlaps(const BufferView& a, const BufferView& b) {
    return a && b && a.buffer == b.buffer && a.offset < b.offset + b.length &&
           b.offset < a.offset + a.length;
}
void validate_buffer(const BufferView& view, const Device& device) {
    require(static_cast<bool>(view) && view.length > 0, "empty GPU resource binding");
    require(view.offset <= view.buffer.size() && view.length <= view.buffer.size() - view.offset,
            "invalid GPU resource span");
    require(view.buffer.implementation()->device_identity == device.implementation().get(),
            "buffer and stream belong to different devices");
    // This is intentionally one-way, even for a later-cancelled recording or
    // graph preparation. Only storage never exposed to a GPU command may skip
    // host synchronization; reads are just as relevant as writes.
    view.buffer.implementation()->fresh_host_storage.store(false, std::memory_order_release);
}
void validate_command(const Command& c, const Device& d) {
    if (c.kind == Command::Kind::Compute) {
        const auto& k = c.compute;
        require(static_cast<bool>(k.kernel), "empty kernel dispatch");
        require(k.kernel.implementation()->device_identity == d.implementation().get(),
                "kernel and stream belong to different devices");
        require(k.grid.x && k.grid.y && k.grid.z && k.group.x && k.group.y && k.group.z,
                "dispatch dimensions must be positive");
        checked_mul(checked_mul(k.grid.x, k.grid.y), k.grid.z);
        auto threads = checked_mul(checked_mul(k.group.x, k.group.y), k.group.z);
        require(threads <= 1024, "threadgroup exceeds portable 1024-thread limit");
        auto word_buffer = [&](const BufferView& view, size_t bytes, const char* message) {
            if (!view)
                return;
            validate_buffer(view, d);
            require(view.offset % 4 == 0 && view.length >= bytes, message);
        };
        word_buffer(k.indirect_groups, 12, "indirect groups require three aligned uint32 values");
        word_buffer(k.predicate, 4, "predicate requires an aligned uint32 value");
        word_buffer(k.dispatch_status, 4, "dispatch status requires an aligned uint32 value");
        require(!k.control || (!k.predicate && !k.indirect_groups),
                "control kernels must be unconditional fixed dispatches");
        require(!k.dispatch_status || k.indirect_groups || k.predicate,
                "dispatch status requires an indirect or predicated dispatch");
        require(!overlaps(k.dispatch_status, k.indirect_groups) &&
                    !overlaps(k.dispatch_status, k.predicate),
                "dispatch status must not overlap the count or predicate input");
        if (k.indirect_groups) {
            require(k.max_groups.x && k.max_groups.y && k.max_groups.z,
                    "indirect dispatch requires explicit positive group-count bounds");
            require(k.max_groups.x <= UINT32_MAX && k.max_groups.y <= UINT32_MAX &&
                        k.max_groups.z <= UINT32_MAX,
                    "indirect group bound exceeds uint32");
            require(checked_mul(k.max_groups.x, k.group.x) <= UINT32_MAX &&
                        checked_mul(k.max_groups.y, k.group.y) <= UINT32_MAX &&
                        checked_mul(k.max_groups.z, k.group.z) <= UINT32_MAX,
                    "indirect grid dimension exceeds uint32");
        }
        require(!k.parallel_group || (!k.indirect_groups && !k.predicate && !k.control),
                "parallel groups support independent fixed compute dispatches only");
        std::vector<uint32_t> buffers, threadgroups, textures, samplers;
        const bool derived =
            k.optional_buffer_mask != UINT32_MAX || !k.pointer_base_arguments.empty();
        const auto materialized =
            derived ? detail::materialize_compute_arguments(k) : std::vector<Argument>{};
        for (const auto& a : derived ? materialized : k.arguments) {
            require(a.index < (a.kind == Argument::Kind::Sampler ? 16u : 31u),
                    "Metal argument index exceeds its portable slot limit");
            auto& used = a.kind == Argument::Kind::Threadgroup ? threadgroups :
                         a.kind == Argument::Kind::Texture ? textures :
                         a.kind == Argument::Kind::Sampler ? samplers : buffers;
            require(std::find(used.begin(), used.end(), a.index) == used.end(),
                    "duplicate GPU argument index");
            used.push_back(a.index);
            if (a.kind == Argument::Kind::Buffer) {
                validate_buffer(a.resource, d);
                require(!overlaps(k.dispatch_status, a.resource),
                        "dispatch status must not overlap payload arguments");
            } else if (a.kind == Argument::Kind::Texture) {
                require(static_cast<bool>(a.image), "empty texture binding");
                require(a.image.implementation()->device_identity == d.implementation().get(),
                        "texture and stream belong to different devices");
                require(a.access == Access::Read, "uploaded images are read-only textures");
            } else if (a.kind == Argument::Kind::Sampler) {
                require(static_cast<bool>(a.sampling), "empty sampler binding");
                require(a.sampling.implementation()->device_identity == d.implementation().get(),
                        "sampler and stream belong to different devices");
            } else if (a.kind == Argument::Kind::Bytes)
                require(!a.value.empty(), "empty constant argument");
            else if (a.kind == Argument::Kind::Threadgroup)
                require(a.threadgroup_bytes > 0 && a.threadgroup_bytes % 16 == 0,
                        "threadgroup allocation must be a nonzero multiple of 16 bytes");
            else {
                require(a.kind == Argument::Kind::NullBuffer, "unknown GPU argument kind");
                require(!a.resource && !a.resource.offset && !a.resource.length &&
                            a.value.empty() && !a.threadgroup_bytes,
                        "null buffer must not contain a resource or value");
            }
        }
    } else {
        validate_buffer(c.destination, d);
        if (c.kind == Command::Kind::Copy) {
            validate_buffer(c.source, d);
            require(c.source.length == c.destination.length,
                    "copy spans must have the same length");
            if (c.source.buffer == c.destination.buffer) {
                size_t x = c.source.offset, y = c.destination.offset, n = c.source.length;
                require(x == y || x + n <= y || y + n <= x,
                        "overlapping GPU copies are undefined; use a separate buffer");
            }
        }
    }
}
void validate_parallel_pair(const ComputeCommand& a, const ComputeCommand& b) {
    const auto a_derived = a.pointer_base_arguments.empty()
                               ? std::vector<Argument>{}
                               : detail::materialize_compute_arguments(a);
    const auto b_derived = b.pointer_base_arguments.empty()
                               ? std::vector<Argument>{}
                               : detail::materialize_compute_arguments(b);
    for (const auto& x : a.pointer_base_arguments.empty() ? a.arguments : a_derived)
        if (x.kind == Argument::Kind::Buffer)
            for (const auto& y : b.pointer_base_arguments.empty() ? b.arguments : b_derived)
                if (y.kind == Argument::Kind::Buffer)
                    require(
                        !overlaps(x.resource, y.resource) ||
                            (x.access == Access::Read && y.access == Access::Read),
                        "parallel commands have overlapping buffer ranges with a declared writer");
}
void validate_sequence(const std::vector<Command>& commands, const Device& device) {
    for (size_t i = 0; i < commands.size(); ++i) {
        validate_command(commands[i], device);
        if (commands[i].kind != Command::Kind::Compute || !commands[i].compute.parallel_group)
            continue;
        for (size_t j = 0; j < i; ++j)
            if (commands[j].kind == Command::Kind::Compute &&
                commands[j].compute.parallel_group == commands[i].compute.parallel_group)
                validate_parallel_pair(commands[j].compute, commands[i].compute);
    }
}

} // namespace metalrt
