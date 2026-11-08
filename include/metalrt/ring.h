#pragma once
#include <metalrt/runtime.h>
#include <functional>
#include <memory>
#include <optional>
#include <type_traits>

namespace metalrt {
namespace detail {
struct RingState;
struct RingPipelineState;
} // namespace detail

struct RingStatistics {
    uint64_t acquisitions = 0;
    uint64_t submissions = 0;
    uint64_t slot_waits = 0;
    uint64_t try_misses = 0;
    uint64_t abandoned_leases = 0;
    uint64_t recoveries = 0;
    uint64_t readbacks = 0;
};

// A producer/consumer ring with one CPU producer lease at a time and multiple
// GPU submissions in flight. Each slot has its own completion. An abandoned
// CPU lease does not advance the schedule. The owning allocation is stable.
// Access/submission through a lease stays on its acquiring thread; move its
// handle within that thread. Alignment is a power of two in the range 1..256.
class RingBuffer {
  public:
    class Lease {
      public:
        Lease() = default;
        Lease(Lease&& other) noexcept;
        Lease& operator=(Lease&& other) noexcept;
        Lease(const Lease&) = delete;
        Lease& operator=(const Lease&) = delete;
        ~Lease() noexcept;
        explicit operator bool() const noexcept { return active_; }
        size_t slot() const;
        size_t size_bytes() const;
        BufferView view() const;
        void* data() const;
        template <class T> T* data_as() const {
            static_assert(std::is_trivially_copyable<T>::value,
                          "mapped slot elements must be trivially copyable");
            auto* pointer = data();
            if (size_bytes() < sizeof(T) || reinterpret_cast<uintptr_t>(pointer) % alignof(T))
                throw std::invalid_argument(
                    "ring slot is too small or misaligned for its mapped type");
            return static_cast<T*>(pointer);
        }
        // Atomically end CPU access and associate the submitted graph's completion.
        // The stream must not already be recording. Invalid stream arguments leave
        // the lease intact; a submission failure poisons the ring until recover().
        Ticket launch(Stream& stream, Graph& graph, const std::vector<EventPoint>& waits = {});
        // One-shot encoding convenience. The callback must use only this stream,
        // must not retain it for later submissions, and must not access mapped slot
        // bytes once encoding begins. Work submitted before a callback exception is
        // still tracked; the ring is poisoned instead of returning that slot to CPU.
        Ticket submit(Stream& stream, const std::function<void(Stream&, BufferView)>& encode);

      private:
        Lease(std::shared_ptr<detail::RingState> state, size_t slot, uint64_t generation);
        void abandon() noexcept;
        void begin_submission(Stream& stream);
        Ticket complete_submission(Ticket ticket);
        void fail_submission(Ticket ticket, std::exception_ptr error) noexcept;
        std::shared_ptr<detail::RingState> state_;
        size_t slot_ = 0;
        uint64_t generation_ = 0;
        bool active_ = false;
        friend class RingBuffer;
    };

    RingBuffer(Device device, size_t slots, size_t bytes_per_slot, size_t alignment = 256);
    RingBuffer(RingBuffer&&) noexcept = default;
    RingBuffer& operator=(RingBuffer&&) noexcept = default;
    RingBuffer(const RingBuffer&) = delete;
    RingBuffer& operator=(const RingBuffer&) = delete;
    ~RingBuffer() = default;
    Lease acquire();                    // Waits only for the scheduled slot.
    std::optional<Lease> try_acquire(); // Returns empty if CPU-owned or GPU-busy.
    void drain();                       // Observe all submitted completions.
    std::vector<uint8_t>
    readback(size_t slot); // Wait/copy one slot without advancing the schedule.
    void recover();        // Explicitly drain failed work and reset scheduling.
    bool poisoned() const;
    size_t slot_count() const;
    size_t bytes_per_slot() const;
    size_t stride_bytes() const;
    Device device() const;
    RingStatistics statistics() const;
    // Bindings are exposed for recording, not for unsynchronized host access.
    // All GPU use of these slots must be submitted through the corresponding lease.
    BufferView binding(size_t slot) const;
    BufferView storage_binding() const;

  private:
    std::optional<Lease> acquire_impl(bool nonblocking);
    void recover_with(const std::function<void()>& reset);
    std::shared_ptr<detail::RingState> state_;
    friend class RingPipeline;
};

enum class RingBindingMode { PerSlot, Indexed };
struct RingBindings {
    BufferView storage;
    BufferView slot_index;  // Indexed mode: one GPU-written uint32_t.
    size_t slot = SIZE_MAX; // PerSlot mode only; Indexed mode uses slot_index.
    size_t slots = 0;
    size_t bytes_per_slot = 0;
    size_t stride_bytes = 0;
    bool indexed() const { return static_cast<bool>(slot_index); }
};

// Owns a private submission stream, stable buffers and prepared replay graphs.
// PerSlot prepares one binding variant per slot. Indexed prepares ONE graph
// whose kernels address the active region of a fixed allocation using a
// GPU-owned slot index. A fixed compute-only Indexed graph uses one ICB under
// Indirect; transfers/dynamic nodes can split it into multiple backend segments.
// Indexed builders must obey that addressing contract and must not retain the
// private Stream for later use.
class RingPipeline {
  public:
    using Builder = std::function<void(Stream&, const RingBindings&)>;
    class Lease {
      public:
        Lease() = default;
        Lease(Lease&&) noexcept = default;
        Lease& operator=(Lease&&) noexcept = default;
        Lease(const Lease&) = delete;
        Lease& operator=(const Lease&) = delete;
        explicit operator bool() const noexcept { return static_cast<bool>(lease_); }
        size_t slot() const { return lease_.slot(); }
        size_t size_bytes() const { return lease_.size_bytes(); }
        BufferView view() const { return lease_.view(); }
        void* data() const { return lease_.data(); }
        template <class T> T* data_as() const { return lease_.template data_as<T>(); }
        Ticket submit(const std::vector<EventPoint>& waits = {});

      private:
        Lease(std::shared_ptr<detail::RingPipelineState> state, RingBuffer::Lease lease);
        std::shared_ptr<detail::RingPipelineState> state_;
        RingBuffer::Lease lease_;
        friend class RingPipeline;
    };
    RingPipeline(Device device, DispatchPolicy policy, size_t slots, size_t bytes_per_slot,
                 RingBindingMode mode, Builder builder);
    RingPipeline(RingPipeline&&) noexcept = default;
    RingPipeline& operator=(RingPipeline&&) noexcept = default;
    RingPipeline(const RingPipeline&) = delete;
    RingPipeline& operator=(const RingPipeline&) = delete;
    ~RingPipeline() = default;
    Lease acquire();
    std::optional<Lease> try_acquire();
    void drain();
    std::vector<uint8_t> readback(size_t slot);
    void recover();
    bool poisoned() const;
    size_t slot_count() const;
    size_t graph_count() const;
    RingBindingMode mode() const;
    RingStatistics ring_statistics() const;
    Statistics stream_statistics() const;
    const Graph& graph(size_t variant = 0) const;

  private:
    std::shared_ptr<detail::RingPipelineState> state_;
};
} // namespace metalrt
