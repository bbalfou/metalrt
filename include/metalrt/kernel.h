#pragma once
// Owned kernel handles and backend-independent command snapshots.
#include <metalrt/buffer.h>

namespace metalrt {
struct Argument {
    enum class Kind { Buffer, Bytes, Threadgroup, NullBuffer };
    uint32_t index = 0;
    Kind kind = Kind::Buffer;
    BufferView resource;
    Access access = Access::Read;
    std::vector<uint8_t> value;
    size_t threadgroup_bytes = 0;
    static Argument buffer(uint32_t index, BufferView resource, Access access = Access::Read);
    // An explicitly absent optional device pointer. Ordinary buffer() still
    // rejects an empty allocation, avoiding accidental missing resources.
    static Argument null_buffer(uint32_t index);
    static Argument bytes(uint32_t index, const void* data, size_t size);
    template <class T> static Argument scalar(uint32_t index, const T& x) {
        static_assert(std::is_trivially_copyable<T>::value,
                      "GPU constants must be trivially copyable");
        return bytes(index, &x, sizeof(T));
    }
    static Argument threadgroup(uint32_t index, size_t bytes);
};
struct ComputeCommand;
using CpuKernel = std::function<void(const ComputeCommand&)>;
class Kernel {
  public:
    Kernel() = default;
    explicit Kernel(std::shared_ptr<detail::KernelState> state) : state_(std::move(state)) {}
    Kernel(std::shared_ptr<detail::KernelState> state, std::shared_ptr<detail::DeviceBackend> owner)
        : state_(std::move(state)), device_owner_(std::move(owner)) {}
    explicit operator bool() const { return static_cast<bool>(state_); }
    const std::string& name() const;
    // Limits of this compiled pipeline, including the stricter limit when an
    // indirect variant exists. CPU reference kernels use the runtime's portable
    // threadgroup ceiling and report no static threadgroup allocation.
    size_t max_threads_per_threadgroup() const;
    size_t static_threadgroup_memory_bytes() const;
    size_t thread_execution_width() const; // Native SIMD width; zero on Reference.
    const std::shared_ptr<detail::KernelState>& implementation() const { return state_; }
    bool operator==(const Kernel& k) const { return state_ == k.state_; }

  private:
    std::shared_ptr<detail::KernelState> state_;
    // Ownership belongs to user/command handles, not to the cached KernelState:
    // a backend owns its cache and must not form a strong reference cycle.
    std::shared_ptr<detail::DeviceBackend> device_owner_;
};
struct ComputeCommand {
    Kernel kernel;
    Size3 grid, group;
    std::vector<Argument> arguments;
    // Commands are sequential by default, including inside an ICB.
    bool barrier_before = true;
    // Empty means fixed grid. Otherwise three uint32_t THREADGROUP counts produced
    // by the GPU, clamped to max_groups before dispatch. Any zero dimension skips.
    BufferView indirect_groups;
    Size3 max_groups{0, 0, 0};
    // Optional uint32_t GPU predicate. Zero suppresses payload dispatch entirely.
    BufferView predicate;
    // Optional uint32_t status: bit 0=clamped, bit 1=predicate inactive.
    BufferView dispatch_status;
    // Runtime control kernels execute unconditionally and explicitly consume a
    // parent predicate. They are counted separately from user payload dispatches.
    bool control = false;
    uint64_t parallel_group = 0; // Nonzero for checked explicit parallel groups.
    // Opt-in shader ABI for optional buffers. A reserved constant uint at this
    // index contains one bit per buffer slot: real buffers are present, while
    // NullBuffer bindings are absent. Native preparation binds real fallback
    // storage for absent arguments; the shader must restore semantic nullptr
    // from this mask before using those pointers. UINT32_MAX disables this ABI.
    uint32_t optional_buffer_mask = UINT32_MAX;
    // Aligned pointer defaults, as (source slot, base slot). Missing bases are
    // derived when preparing each command version, so graph source edits update
    // their hidden base and presence bit. Explicitly bound bases take precedence.
    std::vector<std::pair<uint32_t, uint32_t>> pointer_base_arguments;
    const Argument& argument(uint32_t index) const;
    void*
    reference_buffer(uint32_t index) const; // CPU backend callbacks only; includes Private storage.
    template <class T> T scalar(uint32_t index) const {
        const auto& a = argument(index);
        if (a.kind != Argument::Kind::Bytes || a.value.size() != sizeof(T))
            throw std::invalid_argument("constant ABI mismatch");
        T result;
        std::memcpy(&result, a.value.data(), sizeof(T));
        return result;
    }
    template <class T> T* buffer(uint32_t index) const {
        return static_cast<T*>(reference_buffer(index));
    }
};
struct Command {
    enum class Kind { Compute, Copy, Fill };
    Kind kind = Kind::Compute;
    ComputeCommand compute;
    BufferView source, destination;
    uint8_t fill_value = 0;
};

} // namespace metalrt
