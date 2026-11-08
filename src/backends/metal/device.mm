// Native device capabilities, allocations, and borrowed-queue integration.
#include "device.h"
namespace metalrt {
namespace detail {
namespace metal {
MetalDevice::MetalDevice(id<MTLDevice> device) : device_(device) {
    capabilities_.backend = BackendKind::Metal;
    capabilities_.name = ns_string(device_.name);
    capabilities_.shared_memory = true;
    capabilities_.runtime_compilation = true;
    capabilities_.binary_archives = true;
    if ([device_ supportsCounterSampling:MTLCounterSamplingPointAtStageBoundary]) {
        for (id<MTLCounterSet> set in device_.counterSets) {
            if ([set.name isEqualToString:MTLCommonCounterSetTimestamp])
                capabilities_.kernel_timestamps = true;
            if ([set.name isEqualToString:MTLCommonCounterSetStatistic])
                capabilities_.statistic_counters = true;
            if ([set.name isEqualToString:MTLCommonCounterSetStageUtilization])
                capabilities_.utilization_counters = true;
        }
    }
    capabilities_.max_threads_per_threadgroup = device_.maxThreadsPerThreadgroup.width;
    capabilities_.max_threadgroup_memory_bytes = device_.maxThreadgroupMemoryLength;
    // Feature-family queries describe the actual device; never infer a
    // hardware family from a marketing name or a Reference fixture.
    // Apple publishes stable MTLGPUFamily values. Using the published
    // values keeps hardware detection correct when built with an older SDK
    // whose headers do not yet spell the newer family enumerators.
    constexpr MTLGPUFamily apple9 = static_cast<MTLGPUFamily>(1009);
    constexpr MTLGPUFamily apple10 = static_cast<MTLGPUFamily>(1010);
    if (@available(macOS 26.0, *)) {
        if ([device_ supportsFamily:apple10])
            capabilities_.apple_gpu_family = 10;
    }
    if (@available(macOS 14.0, *)) {
        if (!capabilities_.apple_gpu_family && [device_ supportsFamily:apple9])
            capabilities_.apple_gpu_family = 9;
    }
    for (uint32_t family = 8; !capabilities_.apple_gpu_family && family; --family)
        if ([device_ supportsFamily:static_cast<MTLGPUFamily>(1000 + family)])
            capabilities_.apple_gpu_family = family;
    const bool family =
        [device_ supportsFamily:MTLGPUFamilyApple3] || [device_ supportsFamily:MTLGPUFamilyMac2];
    if (family) {
        MTLIndirectCommandBufferDescriptor* descriptor = [MTLIndirectCommandBufferDescriptor new];
        descriptor.commandTypes = MTLIndirectCommandTypeConcurrentDispatch;
        descriptor.inheritBuffers = NO;
        descriptor.inheritPipelineState = NO;
        descriptor.maxKernelBufferBindCount = 1;
        id<MTLIndirectCommandBuffer> probe =
            [device_ newIndirectCommandBufferWithDescriptor:descriptor
                                            maxCommandCount:1
                                                    options:MTLResourceStorageModeShared |
                                                            MTLResourceHazardTrackingModeTracked];
        capabilities_.indirect_compute = probe != nil;
    }
}

std::shared_ptr<BufferAllocation>
MetalDevice::import_metal_buffer(void* native, std::shared_ptr<void> external_owner) {
    return import_metal_buffer(native, std::move(external_owner), false);
}

std::shared_ptr<BufferAllocation>
MetalDevice::import_metal_buffer(void* native, std::shared_ptr<void> external_owner,
                                 bool allow_untracked) {
    std::lock_guard<std::mutex> lock(mutex_);
    @autoreleasepool {
        id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)native;
        if (!buffer)
            throw std::invalid_argument("cannot import a null Metal buffer");
        if (buffer.device.registryID != device_.registryID)
            throw std::invalid_argument("imported Metal buffer belongs to another device");
        if (buffer.storageMode != MTLStorageModeShared &&
            buffer.storageMode != MTLStorageModePrivate)
            throw std::invalid_argument(
                "Metal buffer import requires Shared or Private storage; Managed storage needs "
                "explicit CPU/GPU coherence and is not represented by this runtime");
        if (buffer.hazardTrackingMode != MTLHazardTrackingModeTracked && !allow_untracked)
            throw std::invalid_argument(
                "Metal buffer import requires tracked resources; untracked external buffers need "
                "an explicit producer/consumer fence contract");
        const void* identity = (__bridge const void*)buffer;
        auto& cached = imported_buffers_[identity];
        auto result = cached.lock();
        if (!result) {
            result = std::make_shared<MetalAllocation>();
            result->buffer = buffer;
            result->bytes = buffer.length;
            result->storage = buffer.storageMode == MTLStorageModeShared ? StorageMode::Shared
                                                                         : StorageMode::Private;
            result->device_identity = this;
            result->device_owner = shared_from_this();
            cached = result;
            // Weak entries are not allocator leases. Bound cache metadata
            // in long-running training with changing temporary tensors.
            if (imported_buffers_.size() > 1024)
                for (auto it = imported_buffers_.begin(); it != imported_buffers_.end();)
                    if (it->second.expired())
                        it = imported_buffers_.erase(it);
                    else
                        ++it;
        }
        if (external_owner)
            result->external_owners.emplace(external_owner.get(), std::move(external_owner));
        return result;
    }
}

std::shared_ptr<BufferAllocation> MetalDevice::allocate(size_t bytes, StorageMode storage) {
    @autoreleasepool {
        if (bytes > device_.maxBufferLength)
            throw std::length_error("allocation exceeds MTLDevice.maxBufferLength");
        // Packed-half atomic updates touch a complete four-byte word at
        // the allocation tail. Keep logical bounds unchanged while giving
        // the native allocation valid backing for that final word.
        const size_t physical_bytes = checked_add(bytes, size_t{3}) & ~size_t{3};
        if (physical_bytes > device_.maxBufferLength)
            throw std::length_error("allocation exceeds MTLDevice.maxBufferLength");
        MTLResourceOptions mode = storage == StorageMode::Private ? MTLResourceStorageModePrivate
                                                                  : MTLResourceStorageModeShared;
        auto result = std::make_shared<MetalAllocation>();
        result->bytes = bytes;
        result->storage = storage;
        result->device_identity = this;
        result->buffer = [device_ newBufferWithLength:std::max<size_t>(physical_bytes, 4)
                                              options:mode | MTLResourceHazardTrackingModeTracked];
        if (!result->buffer)
            metal_error("newBufferWithLength");
        return result;
    }
}

std::unique_ptr<QueueBackend> MetalDevice::make_queue() {
    return make_queue_impl({});
}

std::unique_ptr<QueueBackend>
MetalDevice::make_external_metal_queue(std::function<void*()> provider) {
    if (!provider)
        throw std::invalid_argument("borrowed Metal streams require a command-buffer provider");
    return make_queue_impl(std::move(provider));
}

std::unique_ptr<QueueBackend> MetalDevice::make_queue_impl(std::function<void*()> provider) {
    std::lock_guard<std::mutex> lock(mutex_);
    @autoreleasepool {
        if (!transfers_) {
            auto transfers = std::make_shared<TransferPipelines>();
            id<MTLLibrary> library = library_locked(byte_transfer_source);
            NSError* error = nil;
            id<MTLFunction> copy = [library newFunctionWithName:@"metalrt_copy_bytes"];
            id<MTLFunction> fill = [library newFunctionWithName:@"metalrt_fill_bytes"];
            id<MTLFunction> prepare = [library newFunctionWithName:@"metalrt_prepare_dispatch"];
            if (!copy || !fill || !prepare)
                metal_error("loading internal runtime kernels");
            transfers->copy = [device_ newComputePipelineStateWithFunction:copy error:&error];
            if (!transfers->copy)
                metal_error("compiling byte copy kernel", error);
            error = nil;
            transfers->fill = [device_ newComputePipelineStateWithFunction:fill error:&error];
            if (!transfers->fill)
                metal_error("compiling byte fill kernel", error);
            error = nil;
            transfers->prepare_dispatch = [device_ newComputePipelineStateWithFunction:prepare
                                                                                 error:&error];
            if (!transfers->prepare_dispatch)
                metal_error("compiling GPU dispatch preparation kernel", error);
            transfers_ = std::move(transfers);
        }
        return std::make_unique<MetalQueue>(device_, this, capabilities_.indirect_compute,
                                            transfers_, std::move(provider));
    }
}

std::shared_ptr<EventState> MetalDevice::make_event() {
    @autoreleasepool {
        auto result = std::make_shared<MetalEvent>();
        result->device_identity = this;
        result->event = [device_ newSharedEvent];
        if (!result->event)
            metal_error("newSharedEvent");
        result->event.label = @"metalrt timeline";
        return result;
    }
}

} // namespace metal
} // namespace detail
} // namespace metalrt
