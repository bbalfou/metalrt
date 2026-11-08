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
// Completed Metal command-buffer timestamps, in seconds on the host timebase.
// The sum is a sum of command-buffer intervals, not exclusive ALU busy time.
// The span also includes gaps between buffers. Neither is CPU submission time.
struct GpuTiming {
    bool supported = false;
    uint64_t command_buffers = 0, timed_command_buffers = 0;
    double command_buffer_seconds = 0;
    double first_gpu_start_seconds = 0, last_gpu_end_seconds = 0;
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
// lookups. Source compilation is separate: Metal still needs a library/function
// to describe a pipeline loaded from a persistent binary archive.
struct KernelCacheStatistics {
    uint64_t memory_hits = 0, source_compilations = 0;
    uint64_t archive_hits = 0, archive_misses = 0, archive_writes = 0, archive_errors = 0;
};
struct KernelCompileOptions {
    bool fast_math = false;
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
