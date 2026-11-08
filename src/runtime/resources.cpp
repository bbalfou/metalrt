// Buffer ownership, checked spans, and weak raw-pointer lookup.
#include "runtime_internal.h"

namespace metalrt {
namespace {
std::mutex registry_mutex;
std::map<uintptr_t, std::weak_ptr<detail::BufferAllocation>> registry;
} // namespace
size_t checked_add(size_t a, size_t b) {
    if (b > SIZE_MAX - a)
        throw std::overflow_error("byte size addition overflow");
    return a + b;
}
size_t checked_mul(size_t a, size_t b) {
    if (a && b > SIZE_MAX / a)
        throw std::overflow_error("byte size multiplication overflow");
    return a * b;
}
Buffer::Buffer(std::shared_ptr<detail::BufferAllocation> p) : impl_(std::move(p)) {}
size_t Buffer::size() const {
    return impl_ ? impl_->bytes : 0;
}
StorageMode Buffer::storage_mode() const {
    return impl_ ? impl_->storage : StorageMode::Shared;
}
Device Buffer::device() const {
    if (!impl_ || !impl_->device_owner)
        throw std::logic_error("empty buffer has no owning device");
    return Device{impl_->device_owner};
}
void* Buffer::contents() const {
    if (!impl_)
        return nullptr;
    if (impl_->storage != StorageMode::Shared)
        throw std::logic_error(
            "Private GPU storage is not CPU-addressable; use a Shared staging buffer");
    return impl_->contents();
}
BufferView Buffer::view(size_t offset, size_t bytes) const {
    require(offset <= size(), "buffer view offset out of range");
    return {*this, offset, bytes == SIZE_MAX ? size() - offset : bytes};
}
BufferView::BufferView(Buffer b, size_t o, size_t n) : buffer(std::move(b)), offset(o), length(n) {
    require(o <= buffer.size() && n <= buffer.size() - o, "buffer view out of range");
}
void* BufferView::data() const {
    auto* p = static_cast<uint8_t*>(buffer.contents());
    return p ? p + offset : nullptr;
}
BufferView BufferView::slice(size_t o, size_t n) const {
    require(o <= length, "buffer slice offset out of range");
    if (n == SIZE_MAX)
        n = length - o;
    require(n <= length - o, "buffer slice length out of range");
    return {buffer, offset + o, n};
}
namespace detail {
void register_allocation(const std::shared_ptr<BufferAllocation>& a) {
    if (!a || !a->bytes || a->storage != StorageMode::Shared)
        return;
    auto start = reinterpret_cast<uintptr_t>(a->contents());
    require(start != 0, "Shared allocation has no contents");
    std::lock_guard<std::mutex> lock(registry_mutex);
    // Weak references never keep retired allocations or arena leases alive.
    for (auto i = registry.begin(); i != registry.end();) {
        if (i->second.expired())
            i = registry.erase(i);
        else
            ++i;
    }
    // Repeated imports of the same allocation are harmless. Distinct native
    // buffers that alias host addresses cannot be distinguished by a raw
    // pointer; reject that ambiguity instead of silently changing its owner.
    const auto end = checked_add(start, a->bytes);
    auto next = registry.lower_bound(start);
    for (auto candidate = next; candidate != registry.end() && candidate->first < end;
         ++candidate) {
        auto existing = candidate->second.lock();
        if (existing == a)
            return;
        require(!existing, "Shared allocation overlaps another registered allocation");
    }
    if (next != registry.begin()) {
        const auto previous = std::prev(next);
        if (auto existing = previous->second.lock())
            require(start - previous->first >= existing->bytes,
                    "Shared allocation overlaps another registered allocation");
    }
    registry[start] = a;
}
} // namespace detail
BufferView resolve_pointer(const void* ptr, size_t bytes) {
    if (!ptr && !bytes)
        return {};
    require(ptr != nullptr, "null GPU pointer with a nonzero byte count");
    const auto address = reinterpret_cast<uintptr_t>(ptr);
    std::lock_guard<std::mutex> lock(registry_mutex);
    auto pos = registry.upper_bound(address);
    if (pos == registry.begin())
        throw std::invalid_argument("pointer is not a live runtime Shared allocation");
    --pos;
    auto allocation = pos->second.lock();
    if (!allocation)
        throw std::invalid_argument("pointer refers to a retired GPU allocation");
    size_t offset = address - pos->first;
    require(offset <= allocation->bytes && bytes <= allocation->bytes - offset,
            "GPU pointer span exceeds its allocation");
    // One-past-the-end is permitted only for an empty view, never for a binding.
    return {Buffer{std::move(allocation)}, offset, bytes};
}
Argument Argument::buffer(uint32_t i, BufferView b, Access access) {
    require(static_cast<bool>(b) && b.length > 0,
            "buffer bindings must have a nonempty allocation");
    Argument a;
    a.index = i;
    a.kind = Kind::Buffer;
    a.resource = std::move(b);
    a.access = access;
    return a;
}
Argument Argument::null_buffer(uint32_t i) {
    Argument a;
    a.index = i;
    a.kind = Kind::NullBuffer;
    a.access = Access::ReadWrite;
    return a;
}
Argument Argument::bytes(uint32_t i, const void* p, size_t n) {
    require(p && n, "constant arguments must be nonempty");
    Argument a;
    a.index = i;
    a.kind = Kind::Bytes;
    auto* begin = static_cast<const uint8_t*>(p);
    a.value.assign(begin, begin + n);
    return a;
}
Argument Argument::threadgroup(uint32_t i, size_t n) {
    require(n > 0, "threadgroup memory length must be positive");
    Argument a;
    a.index = i;
    a.kind = Kind::Threadgroup;
    a.threadgroup_bytes = n;
    return a;
}
const Argument& ComputeCommand::argument(uint32_t i) const {
    for (auto& a : arguments)
        if (a.index == i && a.kind != Argument::Kind::Threadgroup)
            return a;
    throw std::out_of_range("kernel argument index is not bound");
}
void* ComputeCommand::reference_buffer(uint32_t i) const {
    const auto& a = argument(i);
    require(a.kind == Argument::Kind::Buffer || a.kind == Argument::Kind::NullBuffer,
            "argument is not a buffer");
    if (a.kind == Argument::Kind::NullBuffer)
        return nullptr;
    require(a.resource.buffer.device().capabilities().backend == BackendKind::Reference,
            "reference_buffer is only available to CPU reference kernels");
    return static_cast<uint8_t*>(a.resource.buffer.implementation()->contents()) +
           a.resource.offset;
}
std::vector<Argument> detail::materialize_compute_arguments(const ComputeCommand& command) {
    auto arguments = command.arguments;
    const bool mask_enabled = command.optional_buffer_mask != UINT32_MAX;
    if (!mask_enabled && command.pointer_base_arguments.empty())
        return arguments;
    if (mask_enabled)
        require(command.optional_buffer_mask < 31, "optional-buffer mask index must be below 31");
    uint32_t reserved = mask_enabled ? (uint32_t{1} << command.optional_buffer_mask) : 0;
    uint32_t derived = 0, sources = 0;
    for (const auto& mapping : command.pointer_base_arguments) {
        const auto source = mapping.first, base = mapping.second;
        require(source < 31 && base < 31 && source != base,
                "pointer-base arguments require distinct buffer slots below 31");
        require(!mask_enabled || source != command.optional_buffer_mask,
                "pointer-base source conflicts with the optional-buffer mask");
        const auto bit = uint32_t{1} << base;
        require(!(reserved & bit), "derived buffer argument indices must be distinct and not "
                                   "overlap the optional-buffer mask");
        reserved |= bit;
        derived |= bit;
        sources |= uint32_t{1} << source;
    }
    require(!(sources & derived), "pointer-base arguments cannot be chained");
    for (const auto& argument : arguments) {
        if (argument.kind == Argument::Kind::Threadgroup)
            continue;
        require(argument.index < 31, "GPU argument index must be below 31");
        require(!mask_enabled || argument.index != command.optional_buffer_mask,
                "kernel argument conflicts with the reserved optional-buffer mask slot");
    }
    arguments.reserve(
        checked_add(arguments.size(),
                    checked_add(command.pointer_base_arguments.size(), mask_enabled ? 1u : 0u)));
    for (const auto& mapping : command.pointer_base_arguments) {
        // Pointer-base mappings fill missing slots, just like the RTC defaults
        // that precede them. An explicit/default base remains caller-owned;
        // only absent bases follow source edits across graph versions.
        if (std::any_of(arguments.begin(), arguments.end(), [&](const Argument& argument) {
                return argument.kind != Argument::Kind::Threadgroup &&
                       argument.index == mapping.second;
            }))
            continue;
        const auto source =
            std::find_if(arguments.begin(), arguments.end(), [&](const Argument& argument) {
                return argument.kind != Argument::Kind::Threadgroup &&
                       argument.index == mapping.first;
            });
        require(source != arguments.end(), "pointer-base source slot is not bound");
        if (source->kind == Argument::Kind::NullBuffer) {
            arguments.push_back(Argument::null_buffer(mapping.second));
        } else {
            require(source->kind == Argument::Kind::Buffer, "pointer-base source must be a buffer");
            const auto offset = source->resource.offset & ~size_t{3};
            const auto length =
                checked_add(source->resource.length, source->resource.offset - offset);
            auto base = source->resource.buffer.view(offset, length);
            arguments.push_back(Argument::buffer(mapping.second, std::move(base), source->access));
        }
    }
    if (mask_enabled) {
        uint32_t present = 0;
        for (const auto& argument : arguments)
            if (argument.kind == Argument::Kind::Buffer)
                present |= uint32_t{1} << argument.index;
        arguments.push_back(Argument::scalar(command.optional_buffer_mask, present));
    }
    return arguments;
}

} // namespace metalrt
