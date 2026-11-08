#pragma once
// Dispatch and recording primitives. Higher-level helpers should build on this layer.
#include <metalrt/completion.h>

namespace metalrt {
class Graph;
class Stream {
  public:
    explicit Stream(Device device = Device::current(),
                    DispatchPolicy policy = DispatchPolicy::Direct);
    Stream(Device device, DispatchPolicy policy, StreamOrdering ordering);
    ~Stream() noexcept;
    Stream(const Stream&) = delete;
    Stream& operator=(const Stream&) = delete;
    // Own another handle to this exact queue/ordering/capture state. Releasing
    // an alias does not synchronize while another client handle exists. The
    // original owner's destructor keeps its existing completion behavior; the
    // last alias retires any work enqueued after that owner was destroyed.
    std::shared_ptr<Stream> retain() const;
    Device device() const;
    DispatchPolicy policy() const;
    void set_policy(DispatchPolicy policy);
    StreamOrdering ordering() const;
    void set_ordering(StreamOrdering ordering);
    void dispatch(Kernel kernel, Size3 grid, Size3 group, std::vector<Argument> args);
    void dispatch_indirect(Kernel kernel, BufferView threadgroup_counts, Size3 max_groups,
                           Size3 group, std::vector<Argument> args, BufferView status = {});
    void enqueue(Command command);
    void copy(BufferView source, BufferView destination);
    void fill(BufferView destination, uint8_t value = 0);
    Scope batch();       // Submits on normal scope exit. Nested batches join their parent.
    Scope transaction(); // A nested recording checkpoint; exceptions discard its entire suffix.
    Scope parallel();    // Explicit independent static-compute group; joins at scope exit.
    Scope predicate_scope(BufferView predicate); // Used by structured GPU control flow.
    BufferView current_predicate() const;
    bool is_capturing() const;
    bool is_recording() const;
    void synchronize();
    void check_errors();
    Ticket last_ticket() const;
    // Includes implicit default-queue dependencies without blocking the host.
    Ticket completion_ticket();
    Statistics statistics() const;
    void reset_statistics();
    // Optional profiling window. Both calls synchronize. No recording may be
    // active; the initiating thread owns the stream until end_gpu_timing().
    // Reference execution returns supported=false, never fabricated GPU time.
    void begin_gpu_timing();
    GpuTiming end_gpu_timing();
    EventPoint signal(TimelineEvent& event);
    Ticket wait(const EventPoint& point); // Encodes a GPU wait, never a CPU wait.
    // Synchronous CPU boundary: flush/wait, execute once. Not allowed in a graph capture.
    void host(const std::function<void()>& function);

  private:
    Stream(Device device, DispatchPolicy policy, StreamOrdering ordering,
           std::unique_ptr<detail::QueueBackend> queue);
    Stream(std::shared_ptr<detail::StreamState> state, bool retained_alias);
    std::shared_ptr<detail::StreamState> state_;
    bool retained_alias_ = false;
    friend class Graph;
    friend struct detail::MetalInterop;
};
Stream& default_stream();
// One Default-role stream per device, shared by all host threads. Ordinary
// default_stream() retains its independent, thread-local runtime behavior.
Stream& ordered_default_stream(Device device = Device::current());
Stream& stream_or_default(Stream* stream);
void synchronize_device(Device device = Device::current());

// Prepare a buffer for direct host access. A fresh runtime allocation that has
// never been bound to a queued/captured GPU command requires no wait, even when
// unrelated external command buffers remain uncommitted. All other storage
// conservatively synchronizes its owning device. Imported allocations never
// take the fresh-storage shortcut. As with contents(), the caller must prevent
// new concurrent submissions while accessing the memory on the host.
void synchronize_buffer(BufferView buffer);

} // namespace metalrt
