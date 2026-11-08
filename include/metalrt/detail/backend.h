#pragma once
// Backend contract shared by native Metal and CPU validation. Public handles own
// snapshots; queues implement preparation/submission without changing recording semantics.
#include <metalrt/runtime.h>
#include <atomic>
#include <mutex>
#include <unordered_map>

namespace metalrt {
namespace detail {
struct BufferAllocation {
    size_t bytes = 0;
    StorageMode storage = StorageMode::Shared;
    const void* device_identity = nullptr;
    std::shared_ptr<DeviceBackend> device_owner;
    // Only Device::allocate enables this one-way optimization. Imports and
    // backend-owned allocations are conservative until proven fresh by that
    // public allocation path. Any queued/captured GPU binding clears it.
    std::atomic<bool> fresh_host_storage{false};
    virtual ~BufferAllocation() = default;
    virtual void* contents() const = 0;
};
struct TextureAllocation {
    TextureDescriptor descriptor;
    const void* device_identity = nullptr;
    std::shared_ptr<DeviceBackend> device_owner;
    virtual ~TextureAllocation() = default;
    virtual const uint8_t* reference_pixels() const {
        throw std::logic_error("texture pixels are only accessible on the Reference backend");
    }
};
struct SamplerState {
    SamplerDescriptor descriptor;
    const void* device_identity = nullptr;
    std::shared_ptr<DeviceBackend> device_owner;
    virtual ~SamplerState() = default;
};
struct KernelState {
    std::string entry;
    const void* device_identity = nullptr;
    CpuKernel reference;
    size_t max_threads_per_threadgroup = 1024, static_threadgroup_memory_bytes = 0;
    size_t thread_execution_width = 0;
    virtual ~KernelState() = default;
};
struct Completion {
    virtual ~Completion() = default;
    // wait()/ready() observe this submission only. Ticket traverses dependency
    // graphs iteratively, so long queue prefixes do not consume call-stack depth.
    // Deferred backends may await attachment when waiting=true; a query must
    // return false immediately while attachment is still pending.
    virtual void wait() = 0;
    virtual bool ready() const = 0;
    virtual bool append_dependencies(std::vector<Ticket>&, bool waiting) {
        (void)waiting;
        return true;
    }
    Completion* deletion_next = nullptr;
};
// Completion DAGs can also be deep when their last owner is released. The
// deleter drains an intrusive per-thread list rather than recursively deleting
// the dependency chain. It performs no allocation and preserves virtual cleanup.
void destroy_completion(Completion*) noexcept;
template <class T, class... Args> std::shared_ptr<T> make_completion(Args&&... args) {
    static_assert(std::is_base_of<Completion, T>::value, "completion type required");
    return std::shared_ptr<T>{new T(std::forward<Args>(args)...),
                              [](T* value) { destroy_completion(value); }};
}
struct Plan {
    virtual ~Plan() = default;
};
struct EventState {
    const void* device_identity = nullptr;
    std::shared_ptr<DeviceBackend> device_owner;
    mutable std::mutex reservation_mutex;
    uint64_t next_value = 0, producer_stream = 0;
    virtual ~EventState() = default;
    virtual uint64_t completed_value() const = 0;
};
struct TimingState {
    mutable std::mutex mutex;
    GpuTiming result;
    GpuTimingOptions options;
    std::vector<std::pair<double, double>> intervals;
    void record(double begin_seconds, double end_seconds, bool successful);
    GpuTiming snapshot() const;
};
struct SubmitOptions {
    bool prepare_only = false;
    std::shared_ptr<TimingState> timing;
    std::vector<EventPoint> waits;
    std::shared_ptr<EventState> signal_event;
    uint64_t signal_value = 0;
    // Additional native signals let an adapter's implicit ordering coexist with
    // an explicit caller-owned event in the same command-buffer submission.
    struct Signal {
        std::shared_ptr<EventState> event;
        uint64_t value = 0;
    };
    std::vector<Signal> signals;
};
struct SubmitResult {
    std::shared_ptr<Completion> completion;
    std::shared_ptr<Plan> plan;
    Statistics statistics;
};
struct QueueBackend {
    virtual ~QueueBackend() = default;
    // An external execution owner commits, observes asynchronous failures, and
    // retires borrowed command buffers. Stream destruction must not wait for a
    // command buffer that this owner has not committed yet.
    virtual bool externally_managed() const noexcept { return false; }
    // Commands are fully validated by runtime; buffers/constant values are owned snapshots.
    // A returned completion MUST retain all referenced allocations until GPU completion.
    // An in-flight cached plan MUST NOT be mutated. A new version may be allocated instead.
    virtual SubmitResult submit(const std::vector<Command>& commands, DispatchPolicy policy,
                                std::shared_ptr<Plan> cached = {},
                                const SubmitOptions& options = {}) = 0;
};
struct DeviceBackend : std::enable_shared_from_this<DeviceBackend> {
    std::atomic<uint64_t> buffer_allocations{0}, allocated_bytes{0}, kernel_compilations{0};
    std::atomic<uint64_t> texture_allocations{0}, texture_uploaded_bytes{0}, sampler_allocations{0};
    std::atomic<uint64_t> kernel_memory_hits{0}, source_compilations{0};
    std::atomic<uint64_t> archive_hits{0}, archive_misses{0}, archive_writes{0}, archive_errors{0};
    virtual ~DeviceBackend() = default;
    virtual Capabilities capabilities() const = 0;
    virtual std::shared_ptr<BufferAllocation> allocate(size_t bytes, StorageMode storage) = 0;
    virtual std::shared_ptr<TextureAllocation> upload_texture(TextureDescriptor, const void*,
                                                              size_t) {
        throw std::runtime_error("texture upload is unavailable on this backend");
    }
    virtual std::shared_ptr<SamplerState> make_sampler(SamplerDescriptor) {
        throw std::runtime_error("samplers are unavailable on this backend");
    }
    virtual std::shared_ptr<KernelState> compile(const std::string& source,
                                                 const std::string& entry, CpuKernel reference) = 0;
    virtual std::shared_ptr<KernelState> compile_cached(const std::string& source,
                                                        const std::string& entry,
                                                        const std::string& cache_directory,
                                                        CpuKernel reference) {
        (void)cache_directory;
        return compile(source, entry, std::move(reference));
    }
    virtual std::shared_ptr<KernelState> load_metallib(const std::string&, const std::string& entry,
                                                       const KernelCompileOptions&,
                                                       CpuKernel reference) {
        if (capabilities().backend != BackendKind::Reference)
            throw std::runtime_error("precompiled libraries are unavailable on this backend");
        return compile({}, entry, std::move(reference));
    }
    virtual std::shared_ptr<KernelState> import_metal_pipeline(void*, const std::string&, void*) {
        throw std::runtime_error("Metal pipeline import is unavailable on this backend");
    }
    virtual void clear_kernel_cache() {}
    virtual std::shared_ptr<KernelState>
    compile_cached_with_options(const std::string& source, const std::string& entry,
                                const std::string& cache_directory,
                                const KernelCompileOptions& options, CpuKernel reference) {
        (void)options;
        return compile_cached(source, entry, cache_directory, std::move(reference));
    }
    virtual std::unique_ptr<QueueBackend> make_queue() = 0;
    virtual void* native_metal_device() const {
        throw std::runtime_error("this device does not expose a native Metal device");
    }
    virtual std::shared_ptr<BufferAllocation> import_metal_buffer(void*, std::shared_ptr<void>) {
        throw std::runtime_error("native Metal buffer import is unavailable on this backend");
    }
    virtual std::shared_ptr<BufferAllocation>
    import_metal_buffer(void* buffer, std::shared_ptr<void> owner, bool allow_untracked) {
        if (allow_untracked)
            throw std::runtime_error(
                "explicitly synchronized Metal buffer import is unavailable on this backend");
        return import_metal_buffer(buffer, std::move(owner));
    }
    virtual std::unique_ptr<QueueBackend> make_external_metal_queue(std::function<void*()>) {
        throw std::runtime_error("borrowing a Metal command buffer is unavailable on this backend");
    }
    virtual std::shared_ptr<EventState> make_event() = 0;
};
std::shared_ptr<DeviceBackend> make_reference_device();
std::shared_ptr<DeviceBackend> make_metal_device();
std::shared_ptr<DeviceBackend> make_metal_device(void* native_device);
bool same_commands(const std::vector<Command>& a, const std::vector<Command>& b);
// Materialize compiler-owned arguments from the current logical bindings.
// NullBuffer remains semantic absence here; only the native backend substitutes
// validation-safe storage, after the presence mask has been derived.
std::vector<Argument> materialize_compute_arguments(const ComputeCommand& command);
void register_allocation(const std::shared_ptr<BufferAllocation>& allocation);
} // namespace detail
} // namespace metalrt
