#pragma once
// Completion tickets, timeline events, and exception-safe recording guards.
#include <metalrt/device.h>

namespace metalrt {
class Ticket {
  public:
    Ticket() = default;
    explicit Ticket(std::shared_ptr<detail::Completion> completion)
        : completion_(std::move(completion)) {}
    void wait() const;
    bool ready() const;
    explicit operator bool() const { return static_cast<bool>(completion_); }

  private:
    std::shared_ptr<detail::Completion> completion_;
};
class TimelineEvent;
class EventPoint {
  public:
    EventPoint() = default;
    uint64_t value() const { return value_; }
    void wait() const { completion_.wait(); }
    bool ready() const { return completion_.ready(); }
    explicit operator bool() const { return static_cast<bool>(state_); }
    const std::shared_ptr<detail::EventState>& implementation() const { return state_; }
    const Ticket& completion() const { return completion_; }

  private:
    std::shared_ptr<detail::EventState> state_;
    uint64_t value_ = 0;
    Ticket completion_;
    std::weak_ptr<detail::OrderedSubmission> pending_submission_;
    friend class Stream;
    friend struct detail::StreamState;
};
// One producer Stream per timeline; distinct producers require distinct events.
// A point owns a reserved signal and its completion. Submission can be deferred
// behind a cancellable queue recording; ready() remains false until it completes,
// and wait() reports a deferred submission or execution failure.
class TimelineEvent {
  public:
    explicit TimelineEvent(Device device = Device::current());
    uint64_t completed_value() const;
    Device device() const;

  private:
    std::shared_ptr<detail::EventState> state_;
    friend class Stream;
};
class Scope {
  public:
    Scope() = default;
    Scope(std::function<void()> finish, std::function<void()> cancel,
          std::function<void(std::exception_ptr)> error);
    Scope(Scope&&) noexcept;
    Scope& operator=(Scope&&) = delete;
    Scope(const Scope&) = delete;
    Scope& operator=(const Scope&) = delete;
    ~Scope() noexcept;
    void finish(); // Explicit error-reporting path; destructor defers errors to Stream.
    void cancel() noexcept;
    explicit operator bool() const { return active_; }

  private:
    std::function<void()> finish_, cancel_;
    std::function<void(std::exception_ptr)> error_;
    int exceptions_ = 0;
    bool active_ = false;
};
} // namespace metalrt
