#pragma once
#include <metalrt/ring.h>
#include <cstring>

namespace metalrt {
// Stable, typed Shared storage for frequently changing execution parameters.
// bind() records the buffer identity once; set()/edit()/launch() coordinate host
// access with tracked GPU use. Submit graphs using launch()/submit(), or explicitly
// take responsibility for synchronization when using the low-level view().
template <class T> class ParameterBuffer {
    static_assert(std::is_trivially_copyable<T>::value, "parameter ABI must be trivially copyable");
    static_assert(std::is_default_constructible<T>::value,
                  "parameter readback requires a default-constructible POD value");
    static_assert(alignof(T) <= 256, "parameter ABI requires unsupported alignment");

  public:
    class Edit {
      public:
        explicit Edit(RingBuffer::Lease lease) : lease_(std::move(lease)) {}
        Edit(Edit&&) noexcept = default;
        Edit& operator=(Edit&&) noexcept = default;
        Edit(const Edit&) = delete;
        Edit& operator=(const Edit&) = delete;
        T& value() { return *lease_.template data_as<T>(); }
        void set(const T& value) { std::memcpy(lease_.data(), &value, sizeof(T)); }
        Ticket launch(Stream& stream, Graph& graph, const std::vector<EventPoint>& waits = {}) {
            return lease_.launch(stream, graph, waits);
        }

      private:
        RingBuffer::Lease lease_;
    };
    explicit ParameterBuffer(Device device = Device::current(), const T& initial = T{})
        : ring_(std::move(device), 1, sizeof(T)) {
        set(initial);
    }
    ParameterBuffer(ParameterBuffer&&) noexcept = default;
    ParameterBuffer& operator=(ParameterBuffer&&) noexcept = default;
    ParameterBuffer(const ParameterBuffer&) = delete;
    ParameterBuffer& operator=(const ParameterBuffer&) = delete;
    Edit edit() { return Edit{ring_.acquire()}; }
    std::optional<Edit> try_edit() {
        auto lease = ring_.try_acquire();
        if (!lease)
            return std::nullopt;
        return Edit{std::move(*lease)};
    }
    void set(const T& value) {
        auto lease = ring_.acquire();
        std::memcpy(lease.data(), &value, sizeof(T));
    }
    T get() {
        auto lease = ring_.acquire();
        T result{};
        std::memcpy(&result, lease.data(), sizeof(T));
        return result;
    }
    Ticket launch(Stream& stream, Graph& graph, const std::vector<EventPoint>& waits = {}) {
        auto lease = ring_.acquire();
        return lease.launch(stream, graph, waits);
    }
    Ticket submit(Stream& stream, Graph& graph, const T& value,
                  const std::vector<EventPoint>& waits = {}) {
        auto lease = ring_.acquire();
        std::memcpy(lease.data(), &value, sizeof(T));
        return lease.launch(stream, graph, waits);
    }
    Argument bind(uint32_t index, Access access = Access::Read) const {
        return Argument::buffer(index, view(), access);
    }
    BufferView view() const { return ring_.binding(0); }
    void drain() { ring_.drain(); }
    void recover() { ring_.recover(); }
    bool poisoned() const { return ring_.poisoned(); }
    Device device() const { return ring_.device(); }

  private:
    RingBuffer ring_;
};
} // namespace metalrt
