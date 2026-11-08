#pragma once
// Retained command sequences, reusable backend plans, and transactional updates.
#include <metalrt/stream.h>

namespace metalrt {
class Graph {
  public:
    Graph();
    ~Graph();
    Graph(const Graph&) = delete;
    Graph& operator=(const Graph&) = delete;
    Graph(Graph&&) noexcept = default;
    Graph& operator=(Graph&&) noexcept = default;
    // Record and submit once on successful completion; later launches replay the snapshot.
    Scope capture_guard(Stream* stream = nullptr);
    // Record only. finish() reports errors immediately; unwinding cancels the capture.
    Scope record_guard(Stream* stream = nullptr); // Record without executing.
    void prepare(Stream* stream = nullptr);       // Build reusable backend plan without execution.
    Ticket launch(Stream* stream = nullptr, const std::vector<EventPoint>& waits = {});
    Graph clone() const; // Copies commands/resources; the clone prepares its own plan cache.
    uint64_t revision() const;
    class Update;
    Update update();
    void reset();
    void schedule_synchronize();
    size_t size() const;
    std::vector<Command> commands() const; // Inspection snapshot with retained resources.
  private:
    std::shared_ptr<detail::GraphState> state_;
    Scope record(Stream* stream, bool execute);
};

// Optimistic, transactional edits. A commit fails if another edit or capture has
// changed the snapshot since this transaction began. In-flight versions survive.
class Graph::Update {
  public:
    Update(Update&&) noexcept = default;
    Update& operator=(Update&&) noexcept = default;
    Update(const Update&) = delete;
    Update& operator=(const Update&) = delete;
    Update& set_buffer(size_t command, uint32_t argument, BufferView buffer);
    Update& set_null_buffer(size_t command, uint32_t argument);
    Update& set_bytes(size_t command, uint32_t argument, const void* bytes, size_t size);
    template <class T> Update& set_scalar(size_t command, uint32_t argument, const T& value) {
        static_assert(std::is_trivially_copyable<T>::value, "constant ABI must be trivial");
        return set_bytes(command, argument, &value, sizeof(T));
    }
    Update& set_grid(size_t command, Size3 grid, Size3 group);
    Update& set_indirect_groups(size_t command, BufferView counts, Size3 maximum);
    Update& set_predicate(size_t command, BufferView predicate);
    Update& rebind(const Buffer& from, const Buffer& to); // Preserve offsets, access and lengths.
    void commit();

  private:
    Update(std::shared_ptr<detail::GraphState>, uint64_t, std::vector<Command>, Device);
    std::shared_ptr<detail::GraphState> state_;
    uint64_t revision_ = 0;
    std::vector<Command> commands_;
    Device device_;
    bool committed_ = false;
    friend class Graph;
};

template <DispatchPolicy Policy> class PolicyStream : public Stream {
  public:
    explicit PolicyStream(Device d = Device::current()) : Stream(std::move(d), Policy) {}
};
inline void launch_linear(Stream* stream, Kernel kernel, size_t count, std::vector<Argument> args,
                          size_t threads = 128) {
    if (count == 0)
        return;
    stream_or_default(stream).dispatch(std::move(kernel), {count, 1, 1}, {threads, 1, 1},
                                       std::move(args));
}
} // namespace metalrt
