// Device selection, kernel loading, caching, and native resource import.
#include "runtime_internal.h"

namespace metalrt {
namespace {
thread_local Device current_device;
}
const std::string& Kernel::name() const {
    if (!state_)
        throw std::logic_error("empty kernel");
    return state_->entry;
}
size_t Kernel::max_threads_per_threadgroup() const {
    if (!state_)
        throw std::logic_error("empty kernel");
    return state_->max_threads_per_threadgroup;
}
size_t Kernel::static_threadgroup_memory_bytes() const {
    if (!state_)
        throw std::logic_error("empty kernel");
    return state_->static_threadgroup_memory_bytes;
}
size_t Kernel::thread_execution_width() const {
    if (!state_)
        throw std::logic_error("empty kernel");
    return state_->thread_execution_width;
}
Device Device::system_default() {
#ifdef METALRT_HAS_METAL
    static Device native{detail::make_metal_device()};
    return native;
#else
    throw std::runtime_error("Metal is unavailable in this build. Select Device::reference() "
                             "explicitly for CPU validation.");
#endif
}
Device Device::reference() {
    static Device reference{detail::make_reference_device()};
    return reference;
}
Device import_metal_device(void* native_device) {
    require(native_device != nullptr, "cannot import a null Metal device");
#ifdef METALRT_HAS_METAL
    return Device{detail::make_metal_device(native_device)};
#else
    throw std::runtime_error("native Metal device import is unavailable in this build");
#endif
}
void* native_metal_device(Device device) {
    require(static_cast<bool>(device), "empty device");
    return device.implementation()->native_metal_device();
}
Buffer import_metal_buffer(Device device, void* native_buffer,
                           std::shared_ptr<void> external_owner) {
    return import_metal_buffer(std::move(device), native_buffer, std::move(external_owner),
                               MetalBufferSynchronization::Tracked);
}
Buffer import_metal_buffer(Device device, void* native_buffer, std::shared_ptr<void> external_owner,
                           MetalBufferSynchronization synchronization) {
    require(static_cast<bool>(device), "empty device");
    require(native_buffer != nullptr, "cannot import a null Metal buffer");
    require(synchronization == MetalBufferSynchronization::Tracked ||
                synchronization == MetalBufferSynchronization::Explicit,
            "unknown Metal buffer synchronization mode");
    auto allocation = device.implementation()->import_metal_buffer(
        native_buffer, std::move(external_owner),
        synchronization == MetalBufferSynchronization::Explicit);
    require(static_cast<bool>(allocation), "Metal buffer import did not return an allocation");
    require(allocation->device_owner == device.implementation(),
            "imported buffer has an invalid owning device");
    detail::register_allocation(allocation);
    return Buffer{std::move(allocation)};
}
Device Device::current() {
    if (!current_device)
        current_device = system_default();
    return current_device;
}
void Device::set_current(Device d) {
    require(static_cast<bool>(d), "cannot select an empty device");
    current_device = std::move(d);
}
Capabilities Device::capabilities() const {
    require(static_cast<bool>(*this), "empty device");
    return impl_->capabilities();
}
Buffer Device::allocate(size_t n, StorageMode storage) const {
    require(static_cast<bool>(*this), "empty device");
    if (n == 0)
        return {};
    auto result = impl_->allocate(n, storage);
    if (!result || result->bytes < n)
        throw std::runtime_error("backend allocation failed");
    result->device_owner = impl_;
    result->fresh_host_storage.store(true, std::memory_order_release);
    ++impl_->buffer_allocations;
    impl_->allocated_bytes += n;
    detail::register_allocation(result);
    return Buffer{std::move(result)};
}
Kernel Device::kernel(const std::string& source, const std::string& entry, CpuKernel cpu) const {
    require(static_cast<bool>(*this), "empty device");
    require(!entry.empty(), "kernel entry is empty");
    return Kernel{impl_->compile(source, entry, std::move(cpu)), impl_};
}
Kernel Device::kernel_cached(const std::string& source, const std::string& entry,
                             const std::string& cache_directory, CpuKernel cpu) const {
    require(static_cast<bool>(*this), "empty device");
    require(!entry.empty(), "kernel entry is empty");
    return Kernel{impl_->compile_cached(source, entry, cache_directory, std::move(cpu)), impl_};
}
Kernel Device::kernel_cached(const std::string& source, const std::string& entry,
                             const std::string& cache_directory,
                             const KernelCompileOptions& options, CpuKernel cpu) const {
    require(static_cast<bool>(*this), "empty device");
    require(!entry.empty(), "kernel entry is empty");
    return Kernel{
        impl_->compile_cached_with_options(source, entry, cache_directory, options, std::move(cpu)),
        impl_};
}
Kernel Device::kernel_file(const std::string& path, const std::string& entry, CpuKernel cpu) const {
    if (!std::filesystem::is_regular_file(path))
        throw std::invalid_argument("shader path is not a regular file: " + path);
    std::ifstream file(path, std::ios::binary);
    if (!file)
        throw std::runtime_error("cannot open shader file: " + path);
    std::string source{std::istreambuf_iterator<char>{file}, std::istreambuf_iterator<char>{}};
    if (file.bad())
        throw std::runtime_error("cannot read shader file: " + path);
    if (source.empty())
        throw std::invalid_argument("shader file is empty: " + path);
    return kernel(source, entry, std::move(cpu));
}
DeviceStatistics Device::statistics() const {
    require(static_cast<bool>(*this), "empty device");
    return {impl_->buffer_allocations.load(), impl_->allocated_bytes.load(),
            impl_->kernel_compilations.load()};
}
KernelCacheStatistics Device::kernel_cache_statistics() const {
    require(static_cast<bool>(*this), "empty device");
    return {impl_->kernel_memory_hits.load(), impl_->source_compilations.load(),
            impl_->archive_hits.load(),       impl_->archive_misses.load(),
            impl_->archive_writes.load(),     impl_->archive_errors.load()};
}
void Device::clear_kernel_cache() const {
    require(static_cast<bool>(*this), "empty device");
    impl_->clear_kernel_cache();
}

} // namespace metalrt
