#pragma once
// Retained allocation handles and bounds-checked views; host access never waits implicitly.
#include <metalrt/types.h>

namespace metalrt {
class Buffer {
  public:
    Buffer() = default;
    explicit Buffer(std::shared_ptr<detail::BufferAllocation> impl);
    size_t size() const;
    StorageMode storage_mode() const;
    Device device() const;
    void* contents() const; // Throws for Private memory. Never synchronizes.
    template <class T> T* data() const { return static_cast<T*>(contents()); }
    explicit operator bool() const { return static_cast<bool>(impl_); }
    BufferView view(size_t offset = 0, size_t bytes = SIZE_MAX) const;
    long use_count() const { return impl_.use_count(); }
    const std::shared_ptr<detail::BufferAllocation>& implementation() const { return impl_; }
    bool operator==(const Buffer& b) const { return impl_ == b.impl_; }
    bool operator!=(const Buffer& b) const { return !(*this == b); }

  private:
    std::shared_ptr<detail::BufferAllocation> impl_;
};
class BufferView {
  public:
    Buffer buffer;
    size_t offset = 0, length = 0;
    BufferView() = default;
    BufferView(Buffer b, size_t offset, size_t bytes);
    void* data() const;
    template <class T> T* data_as() const { return static_cast<T*>(data()); }
    BufferView slice(size_t relative_offset, size_t bytes = SIZE_MAX) const;
    explicit operator bool() const { return static_cast<bool>(buffer); }
    bool operator==(const BufferView& b) const {
        return buffer == b.buffer && offset == b.offset && length == b.length;
    }
};

// Raw-pointer compatibility is ONLY for live Shared buffers allocated or
// explicitly imported by this runtime. Private imports require BufferViews.
// Resolving captures a strong allocation reference, including an interior byte offset.
BufferView resolve_pointer(const void* pointer, size_t bytes);
size_t checked_add(size_t a, size_t b);
size_t checked_mul(size_t a, size_t b);

} // namespace metalrt
