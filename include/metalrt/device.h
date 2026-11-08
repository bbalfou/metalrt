#pragma once
// Device selection, memory allocation, and kernel compilation.
#include <metalrt/kernel.h>

namespace metalrt {
class Device {
  public:
    Device() = default;
    explicit Device(std::shared_ptr<detail::DeviceBackend> backend) : impl_(std::move(backend)) {}
    static Device system_default(); // Metal on Apple; throws elsewhere (no silent CPU fallback).
    static Device reference();      // Explicit deterministic CPU validation backend.
    static Device current();
    static void set_current(Device device);
    Capabilities capabilities() const;
    Buffer allocate(size_t bytes, StorageMode storage = StorageMode::Shared) const;
    // Copies host pixels and completes a one-time upload before returning.
    // Metal uses a Private, shader-read texture (not a linear buffer view).
    // bytes_per_row=0 selects tightly packed rows; size includes the last row's pixels.
    // Images are immutable: create another texture to change the pixels.
    Texture2D upload_texture(TextureDescriptor descriptor, const void* pixels, size_t size,
                             size_t bytes_per_row = 0) const;
    Sampler sampler(SamplerDescriptor descriptor = {}) const;
    Kernel kernel(const std::string& source, const std::string& entry,
                  CpuKernel reference = {}) const;
    Kernel kernel(const std::string& source, const std::string& entry,
                  const KernelCompileOptions& options, CpuKernel reference = {}) const;
    // Load an offline-compiled Metal library; never compiles MSL source.
    // Reference execution uses the explicit callback, as for kernel().
    Kernel kernel_metallib(const std::string& path, const std::string& entry,
                           const KernelCompileOptions& options = {},
                           CpuKernel reference = {}) const;
    // Compile through the same backend and retain native GPU binaries between
    // processes. An empty directory disables persistent caching. Cache errors
    // fall back to ordinary compilation and are observable in the statistics.
    Kernel kernel_cached(const std::string& source, const std::string& entry,
                         const std::string& cache_directory, CpuKernel reference = {}) const;
    Kernel kernel_cached(const std::string& source, const std::string& entry,
                         const std::string& cache_directory, const KernelCompileOptions& options,
                         CpuKernel reference = {}) const;
    Kernel kernel_file(const std::string& path, const std::string& entry,
                       CpuKernel reference = {}) const;
    Kernel kernel_file(const std::string& path, const std::string& entry,
                       const KernelCompileOptions& options, CpuKernel reference = {}) const;
    DeviceStatistics statistics() const;
    KernelCacheStatistics kernel_cache_statistics() const;
    // Releases only cache ownership. Recorded/in-flight commands retain their
    // kernels; files on disk are unchanged. Safe to call during execution.
    void clear_kernel_cache() const;
    const std::shared_ptr<detail::DeviceBackend>& implementation() const { return impl_; }
    explicit operator bool() const { return static_cast<bool>(impl_); }
    bool operator==(const Device& d) const { return impl_ == d.impl_; }
    bool operator!=(const Device& d) const { return !(*this == d); }

  private:
    std::shared_ptr<detail::DeviceBackend> impl_;
};

} // namespace metalrt
