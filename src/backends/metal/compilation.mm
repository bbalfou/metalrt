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
    return compile_impl(source, entry, cache_directory, options, std::move(reference), false);
}
std::shared_ptr<KernelState> MetalDevice::load_metallib(const std::string& bytes,
                                                        const std::string& entry,
                                                        const KernelCompileOptions& options,
                                                        CpuKernel reference) {
    return compile_impl(bytes, entry, {}, options, std::move(reference), true);
}
std::shared_ptr<KernelState> MetalDevice::compile_impl(
    const std::string& source, const std::string& entry, const std::string& cache_directory,
    const KernelCompileOptions& requested_options, CpuKernel reference, bool precompiled) {
    const auto options = validated_compile_options(requested_options);
    // Length-prefix the complete input, including its kind and specialization.
    const auto identity = std::string(precompiled ? "library:" : "source:") +
                          std::to_string(source.size()) + ":" + source;
    std::lock_guard<std::mutex> lock(mutex_);
    @autoreleasepool {
        const std::string key = compile_options_key(options) + identity + entry;
        auto found = kernels_.find(key);
        if (found != kernels_.end()) {
            ++kernel_memory_hits;
            auto kernel = found->second;
            // Caching may be enabled after this pipeline was first used.
            // Persist it on that first request without replacing the live
            // pipeline or invalidating any recorded command.
            if (!cache_directory.empty() && !kernel->archive_locations.count(cache_directory)) {
                auto archive = archive_locked(identity, entry, cache_directory, options);
                archive_kernel_locked(*kernel, archive);
                if (store_archive_locked(archive))
                    kernel->archive_locations.insert(cache_directory);
            }
            return kernel;
        }
        auto archive = archive_locked(identity, entry, cache_directory, options);
        id<MTLLibrary> library = library_locked(source, options.fast_math, precompiled);
        NSString* name = [[NSString alloc] initWithBytes:entry.data()
                                                  length:entry.size()
                                                encoding:NSUTF8StringEncoding];
        if (!name)
            throw std::invalid_argument("MSL entry point is not valid UTF-8");
        NSError* specialization_error = nil;
        id<MTLFunction> function = nil;
        auto base = [library newFunctionWithName:name];
        if (!base)
            throw std::runtime_error("MSL entry point not found: " + entry);
        for (MTLFunctionConstant* declaration in base.functionConstantsDictionary.allValues) {
            if (declaration.required &&
                std::none_of(options.constants.begin(), options.constants.end(),
                             [&](const auto& c) { return c.index == declaration.index; }))
                throw std::invalid_argument("missing required function constant: " +
                                            ns_string(declaration.name));
        }
        if (options.constants.empty() && base.functionConstantsDictionary.count == 0) {
            function = base;
        } else {
            MTLFunctionConstantValues* values = [MTLFunctionConstantValues new];
            for (const auto& c : options.constants) {
                MTLDataType type = MTLDataTypeUInt;
                switch (c.type) {
                case FunctionConstant::Type::Bool:
                    type = MTLDataTypeBool;
                    break;
                case FunctionConstant::Type::Int32:
                    type = MTLDataTypeInt;
                    break;
                case FunctionConstant::Type::UInt32:
                    break;
                case FunctionConstant::Type::Float32:
                    type = MTLDataTypeFloat;
                    break;
                }
                // Metal ignores unknown indices; reject them to catch ABI mistakes.
                MTLFunctionConstant* declaration = nil;
                for (MTLFunctionConstant* candidate in base.functionConstantsDictionary.allValues)
                    if (candidate.index == c.index)
                        declaration = candidate;
                if (!declaration || declaration.type != type)
                    throw std::invalid_argument(
                        "function constant index/type does not match shader");
                [values setConstantValue:c.value.data() type:type atIndex:c.index];
            }
            function = [library newFunctionWithName:name
                                     constantValues:values
                                              error:&specialization_error];
            if (!function)
                metal_error("specializing " + entry, specialization_error);
        }
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

std::shared_ptr<KernelState>
MetalDevice::import_metal_pipeline(void* direct, const std::string& name, void* indirect) {
    @autoreleasepool {
        auto kernel = std::make_shared<MetalKernel>();
        kernel->direct = (__bridge id<MTLComputePipelineState>)direct;
        kernel->indirect = (__bridge id<MTLComputePipelineState>)indirect;
        for (auto pipeline : {kernel->direct, kernel->indirect})
            if (pipeline && pipeline.device.registryID != device_.registryID)
                throw std::invalid_argument("imported pipeline belongs to another Metal device");
        if (kernel->indirect && !kernel->indirect.supportIndirectCommandBuffers)
            throw std::invalid_argument("imported indirect pipeline does not support ICBs");
        kernel->entry = name;
        kernel->device_identity = this;
        kernel->indirect_error = "no indirect pipeline supplied at import";
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

id<MTLLibrary> MetalDevice::library_locked(const std::string& source, bool fast_math,
                                           bool precompiled) {
    const std::string key = std::string(precompiled ? "binary:" : "source:") +
                            std::string(1, fast_math ? '\1' : '\0') + source;
    auto found = libraries_.find(key);
    if (found != libraries_.end())
        return found->second->library;
    NSError* error = nil;
    auto holder = std::make_shared<LibraryHolder>();
    if (precompiled) {
        void* copy = std::malloc(source.size());
        if (!copy)
            throw std::bad_alloc();
        std::memcpy(copy, source.data(), source.size());
        dispatch_data_t data =
            dispatch_data_create(copy, source.size(), nullptr, DISPATCH_DATA_DESTRUCTOR_DEFAULT);
        holder->library = [device_ newLibraryWithData:data error:&error];
        if (!holder->library)
            metal_error("loading precompiled Metal library", error);
    } else {
        NSString* code = [[NSString alloc] initWithBytes:source.data()
                                                  length:source.size()
                                                encoding:NSUTF8StringEncoding];
        if (!code)
            throw std::invalid_argument("MSL source is not valid UTF-8");
        MTLCompileOptions* options = [MTLCompileOptions new];
        options.fastMathEnabled = fast_math ? YES : NO;
        ++source_compilations;
        holder->library = [device_ newLibraryWithSource:code options:options error:&error];
        if (!holder->library)
            metal_error("compiling MSL source", error);
    }
    libraries_[key] = holder;
    return holder->library;
}

} // namespace metal
} // namespace detail
} // namespace metalrt
