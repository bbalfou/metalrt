#pragma once
// Portable command dimensions, capabilities, statistics, and forward declarations.
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace metalrt {
// Encoding strategy, independent of whether commands were captured or submitted immediately.
// Indirect reuses static compute segments; dynamic dispatches and transfers are encoded separately.
enum class DispatchPolicy { PerKernel, Direct, Indirect };
enum class BackendKind { Metal, Reference };
enum class StorageMode { Shared, Private };
enum class Access { Read, Write, ReadWrite };
// Optional ordering domain used by adapters with an implicit device-wide queue.
// Independent queues have no implicit dependencies. Blocking queues order with
// the Default queue; distinct Blocking queues remain independent of each other.
enum class StreamOrdering { Independent, Blocking, Default };
// A blocking CPU wait cannot complete until this same thread finishes or
// cancels a recording. The recording and its submitted dependents stay valid.
class RecordingDependencyError : public std::logic_error {
  public:
    using std::logic_error::logic_error;
};
struct Size3 {
    size_t x = 1, y = 1, z = 1;
};
struct Statistics {
    uint64_t command_buffers = 0, compute_dispatches = 0, blit_commands = 0;
    uint64_t icb_builds = 0, icb_reuses = 0, icb_updates = 0;
    uint64_t control_dispatches = 0, event_waits = 0, event_signals = 0, prepare_calls = 0;
    uint64_t cpu_submit_nanoseconds = 0, stream_wait_calls = 0, cpu_wait_nanoseconds = 0;
    Statistics& operator+=(const Statistics& other);
};
enum class GpuCounterSet { None, Statistic, StageUtilization };
struct GpuTimingOptions {
    bool sample_kernels = false;
    GpuCounterSet counters = GpuCounterSet::None;
};
struct GpuCounterValue {
    std::string name;
    uint64_t begin = UINT64_MAX, end = UINT64_MAX;
    bool valid() const { return begin != UINT64_MAX && end != UINT64_MAX; }
};
struct KernelTiming {
    std::string kernel;
    size_t command_index = 0; // Index within its submitted batch/graph.
    bool valid = false;
    double seconds = 0;
    uint64_t gpu_start = UINT64_MAX, gpu_end = UINT64_MAX; // Raw GPU clock.
    std::vector<GpuCounterValue> counters;
};
// Completed Metal command-buffer timestamps, in seconds on the host timebase.
// The sum counts overlap; the span includes gaps. Neither is CPU submission time.
struct GpuTiming {
    bool supported = false;
    uint64_t command_buffers = 0, timed_command_buffers = 0;
    double command_buffer_seconds = 0;
    double first_gpu_start_seconds = 0, last_gpu_end_seconds = 0;
    // Union of completed command-buffer intervals: overlap counted once.
    // This measures occupied intervals, not exclusive shader/ALU busy time.
    double occupied_seconds = 0;
    std::vector<KernelTiming> kernels;
    bool valid() const {
        return supported && command_buffers > 0 && timed_command_buffers == command_buffers;
    }
    double span_seconds() const {
        return timed_command_buffers ? last_gpu_end_seconds - first_gpu_start_seconds : 0;
    }
};
struct DeviceStatistics {
    uint64_t buffer_allocations = 0, allocated_bytes = 0, kernel_compilations = 0;
    uint64_t texture_allocations = 0, texture_uploaded_bytes = 0, sampler_allocations = 0;
};
// Memory hits count returned kernel handles; archive hits/misses count pipeline
// lookups. Source compilation is separate. Source-based archive lookups still
// need a library/function; kernel_metallib() and pipeline import avoid MSL compilation.
struct KernelCacheStatistics {
    uint64_t memory_hits = 0, source_compilations = 0;
    uint64_t archive_hits = 0, archive_misses = 0, archive_writes = 0, archive_errors = 0;
};
// Scalar Metal function constants, addressed by [[function_constant(index)]].
struct FunctionConstant {
    enum class Type { Bool, Int32, UInt32, Float32 };
    uint32_t index = 0;
    Type type = Type::UInt32;
    std::vector<uint8_t> value;
    template <class T> static FunctionConstant scalar(uint32_t index, T value) {
        static_assert(std::is_same<T, bool>::value || std::is_same<T, int32_t>::value ||
                          std::is_same<T, uint32_t>::value || std::is_same<T, float>::value,
                      "function constants support bool, int32_t, uint32_t and float");
        FunctionConstant result;
        result.index = index;
        if constexpr (std::is_same<T, bool>::value)
            result.type = Type::Bool;
        else if constexpr (std::is_same<T, int32_t>::value)
            result.type = Type::Int32;
        else if constexpr (std::is_same<T, float>::value)
            result.type = Type::Float32;
        result.value.resize(sizeof(T));
        std::memcpy(result.value.data(), &value, sizeof(T));
        return result;
    }
};
struct KernelCompileOptions {
    bool fast_math = false;
    std::vector<FunctionConstant> constants;
};
struct Capabilities {
    BackendKind backend = BackendKind::Reference;
    std::string name;
    bool indirect_compute = false;
    bool shared_memory = true;
    // Highest Apple GPU family reported by Metal's feature query. Zero means
    // no reported Apple family (including the explicit Reference backend).
    uint32_t apple_gpu_family = 0;
    bool runtime_compilation = false;
    bool binary_archives = false;
    // Device-wide ceilings, not a promise that every compiled kernel can use
    // that many threads. Native submission additionally checks pipeline limits.
    size_t max_threads_per_threadgroup = 0;
    size_t max_threadgroup_memory_bytes = 0;
    bool kernel_timestamps = false, statistic_counters = false, utilization_counters = false;
};
namespace detail {
struct BufferAllocation;
struct TextureAllocation;
struct SamplerState;
struct KernelState;
struct DeviceBackend;
struct QueueBackend;
struct Completion;
struct Plan;
struct StreamState;
struct GraphState;
struct MetalInterop;
struct EventState;
struct OrderedSubmission;
} // namespace detail
class BufferView;
class Device;
} // namespace metalrt
