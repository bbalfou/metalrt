#pragma once
// Native device, resource import, and per-device compilation caches.
#include "queue.h"
namespace metalrt {
namespace detail {
namespace metal {
class MetalDevice final : public DeviceBackend {
  public:
    explicit MetalDevice(id<MTLDevice> device);
    Capabilities capabilities() const override { return capabilities_; }
    void* native_metal_device() const override { return (__bridge void*)device_; }
    std::shared_ptr<BufferAllocation>
    import_metal_buffer(void* native, std::shared_ptr<void> external_owner) override;
    std::shared_ptr<BufferAllocation> import_metal_buffer(void* native,
                                                          std::shared_ptr<void> external_owner,
                                                          bool allow_untracked) override;
    std::shared_ptr<BufferAllocation> allocate(size_t bytes, StorageMode storage) override;
    std::shared_ptr<TextureAllocation>
    upload_texture(TextureDescriptor, const void*, size_t bytes_per_row) override;
    std::shared_ptr<SamplerState> make_sampler(SamplerDescriptor) override;
    std::shared_ptr<KernelState> compile(const std::string& source, const std::string& entry,
                                         CpuKernel reference) override;
    std::shared_ptr<KernelState> compile_cached(const std::string& source, const std::string& entry,
                                                const std::string& cache_directory,
                                                CpuKernel reference) override;
    std::shared_ptr<KernelState> compile_cached_with_options(const std::string& source,
                                                             const std::string& entry,
                                                             const std::string& cache_directory,
                                                             const KernelCompileOptions& options,
                                                             CpuKernel reference) override;
    void clear_kernel_cache() override;
    std::unique_ptr<QueueBackend> make_queue() override;
    std::unique_ptr<QueueBackend>
    make_external_metal_queue(std::function<void*()> provider) override;
    std::unique_ptr<QueueBackend> make_queue_impl(std::function<void*()> provider);
    std::shared_ptr<EventState> make_event() override;

  private:
    id<MTLDevice> device_;
    Capabilities capabilities_;
    std::mutex mutex_;
    std::unordered_map<std::string, std::shared_ptr<LibraryHolder>> libraries_;
    std::unordered_map<std::string, std::shared_ptr<MetalKernel>> kernels_;
    std::unordered_map<const void*, std::weak_ptr<MetalAllocation>> imported_buffers_;
    std::shared_ptr<TransferPipelines> transfers_;
    NativeArchive archive_locked(const std::string& source, const std::string& entry,
                                 const std::string& directory, const KernelCompileOptions& options);
    id<MTLComputePipelineState> pipeline_locked(MTLComputePipelineDescriptor* descriptor,
                                                NativeArchive& archive, NSError** error,
                                                MTLComputePipelineReflection** reflection);
    void archive_kernel_locked(const MetalKernel& kernel, NativeArchive& archive);
    bool store_archive_locked(NativeArchive& archive);
    id<MTLLibrary> library_locked(const std::string& source, bool fast_math = false);
};
} // namespace metal
} // namespace detail
} // namespace metalrt
