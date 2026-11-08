#pragma once
#include <metalrt/runtime.h>

namespace metalrt {
// Conditions read one aligned uint32_t from GPU-accessible storage. Comparisons
// are unsigned; a kernel may produce the value immediately before the condition.
enum class Compare : uint32_t {
    NonZero,
    Zero,
    Equal,
    NotEqual,
    Less,
    LessEqual,
    Greater,
    GreaterEqual
};
struct Predicate {
    BufferView value;
    Compare comparison = Compare::NonZero;
    uint32_t reference = 0;
};

// Record both compute-only bodies once. At replay the condition is snapshotted
// once at branch entry, and only the selected body's payloads receive work.
// Nested helpers include their parent's GPU predicate. Recording failures roll
// back the whole helper, including when it joins an enclosing graph or batch.
void if_else(Stream& stream, Predicate condition, const std::function<void()>& then_body,
             const std::function<void()>& else_body = {});

struct LoopStatus {
    uint32_t iterations = 0;
    bool limit_reached = false;
};
class LoopResult {
  public:
    LoopResult() = default;
    BufferView iterations() const;
    BufferView limit_reached() const;
    // Explicit CPU boundary, after the last replay that writes this result. The
    // caller must not submit another writer until the readback has completed.
    LoopStatus read(const Ticket& last_replay) const;

  private:
    explicit LoopResult(Buffer state) : state_(std::move(state)) {}
    Buffer state_;
    friend LoopResult repeat_while(Stream&, Predicate, uint32_t,
                                   const std::function<void(uint32_t)>&);
};

// Record at most max_iterations copies of a compute body. Each iteration checks
// the GPU condition before its payloads; once false, the loop remains inactive.
// The callback receives the recording-time iteration index, not a CPU readback.
// limit_reached is true only if the condition still holds after the last allowed
// iteration. Control dispatches still execute for inactive recorded iterations.
LoopResult repeat_while(Stream& stream, Predicate condition, uint32_t max_iterations,
                        const std::function<void(uint32_t)>& body);
} // namespace metalrt
