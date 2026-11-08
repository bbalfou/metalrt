// MSL library and pipeline compilation with in-memory reuse.
#include "device.h"
namespace metalrt {
namespace detail {
namespace metal {
std::shared_ptr<KernelState> MetalDevice::compile(const std::string& source,
                                                  const std::string& entry, CpuKernel reference) {
    return compile_cached(source, entry, {}, std::move(reference));
}

std::shared_ptr<KernelState> MetalDevice::compile_cached(const std::string& source,
                                                         const std::string& entry,
                                                         const std::string& cache_directory,
                                                         CpuKernel reference) {
    return compile_cached_with_options(source, entry, cache_directory, KernelCompileOptions{},
                                       std::move(reference));
}

std::shared_ptr<KernelState>
MetalDevice::compile_cached_with_options(const std::string& source, const std::string& entry,
                                         const std::string& cache_directory,
                                         const KernelCompileOptions& options, CpuKernel reference) {
    std::lock_guard<std::mutex> lock(mutex_);
    @autoreleasepool {
        const std::string key =
            std::string(1, options.fast_math ? '\1' : '\0') + source + std::string(1, '\0') + entry;
        auto found = kernels_.find(key);
        if (found != kernels_.end()) {
            ++kernel_memory_hits;
            auto kernel = found->second;
            // Caching may be enabled after this pipeline was first used.
            // Persist it on that first request without replacing the live
            // pipeline or invalidating any recorded command.
            if (!cache_directory.empty() && !kernel->archive_locations.count(cache_directory)) {
                auto archive = archive_locked(source, entry, cache_directory, options);
                archive_kernel_locked(*kernel, archive);
                if (store_archive_locked(archive))
                    kernel->archive_locations.insert(cache_directory);
            }
            return kernel;
        }
        auto archive = archive_locked(source, entry, cache_directory, options);
        id<MTLLibrary> library = library_locked(source, options.fast_math);
        NSString* name = [[NSString alloc] initWithBytes:entry.data()
                                                  length:entry.size()
                                                encoding:NSUTF8StringEncoding];
        if (!name)
            throw std::invalid_argument("MSL entry point is not valid UTF-8");
        id<MTLFunction> function = [library newFunctionWithName:name];
        if (!function)
            throw std::runtime_error("MSL entry point not found: " + entry);
        auto kernel = std::make_shared<MetalKernel>();
        kernel->entry = entry;
        kernel->device_identity = this;
        kernel->reference = std::move(reference);
        kernel->function = function;
        NSError* error = nil;
        MTLComputePipelineReflection* reflection = nil;
        MTLComputePipelineDescriptor* descriptor = [MTLComputePipelineDescriptor new];
        descriptor.computeFunction = function;
        descriptor.label = name;
        descriptor.supportIndirectCommandBuffers = NO;
        kernel->direct = pipeline_locked(descriptor, archive, &error, &reflection);
        if (!kernel->direct)
            metal_error("compiling direct pipeline " + entry, error);
        kernel->direct_reflected = reflect_buffers(reflection, kernel->direct_active_buffers,
                                                   kernel->minimum_buffer_bytes);
        if (capabilities_.indirect_compute) {
            descriptor.supportIndirectCommandBuffers = YES;
            error = nil;
            reflection = nil;
            kernel->indirect = pipeline_locked(descriptor, archive, &error, &reflection);
            if (!kernel->indirect)
                kernel->indirect_error = ns_string(error.localizedDescription);
            else
                kernel->indirect_reflected = reflect_buffers(
                    reflection, kernel->indirect_active_buffers, kernel->minimum_buffer_bytes);
        } else
            kernel->indirect_error = "device does not support compute ICBs";
        kernel->max_threads_per_threadgroup = kernel->direct.maxTotalThreadsPerThreadgroup;
        kernel->static_threadgroup_memory_bytes = kernel->direct.staticThreadgroupMemoryLength;
        kernel->thread_execution_width = kernel->direct.threadExecutionWidth;
        if (kernel->indirect) {
            kernel->max_threads_per_threadgroup =
                std::min(kernel->max_threads_per_threadgroup,
                         size_t(kernel->indirect.maxTotalThreadsPerThreadgroup));
            kernel->static_threadgroup_memory_bytes =
                std::max(kernel->static_threadgroup_memory_bytes,
                         size_t(kernel->indirect.staticThreadgroupMemoryLength));
        }
        if (store_archive_locked(archive))
            kernel->archive_locations.insert(cache_directory);
        kernels_[key] = kernel;
        kernel_compilations.fetch_add(1, std::memory_order_relaxed);
        return kernel;
    }
}

void MetalDevice::clear_kernel_cache() {
    std::lock_guard<std::mutex> lock(mutex_);
    // Both maps own native objects strongly. Other Kernel handles, graph
    // commands, queue plans and in-flight completions keep their copies.
    kernels_.clear();
    libraries_.clear();
}

id<MTLLibrary> MetalDevice::library_locked(const std::string& source, bool fast_math) {
    const std::string key = std::string(1, fast_math ? '\1' : '\0') + source;
    auto found = libraries_.find(key);
    if (found != libraries_.end())
        return found->second->library;
    NSString* code = [[NSString alloc] initWithBytes:source.data()
                                              length:source.size()
                                            encoding:NSUTF8StringEncoding];
    if (!code)
        throw std::invalid_argument("MSL source is not valid UTF-8");
    MTLCompileOptions* options = [MTLCompileOptions new];
    options.fastMathEnabled = fast_math ? YES : NO;
    NSError* error = nil;
    auto holder = std::make_shared<LibraryHolder>();
    ++source_compilations;
    holder->library = [device_ newLibraryWithSource:code options:options error:&error];
    if (!holder->library)
        metal_error("compiling MSL source", error);
    libraries_[key] = holder;
    return holder->library;
}

} // namespace metal
} // namespace detail
} // namespace metalrt
