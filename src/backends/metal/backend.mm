// Native backend factories share a backend by the physical device registry ID.
#include "device.h"
namespace metalrt {
namespace detail {
using namespace metal;
std::shared_ptr<DeviceBackend> make_metal_device() {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device)
            throw std::runtime_error("No Metal-capable device is available");
        return make_metal_device((__bridge void*)device);
    }
}
std::shared_ptr<DeviceBackend> make_metal_device(void* native) {
    @autoreleasepool {
        id<MTLDevice> device = (__bridge id<MTLDevice>)native;
        if (!device)
            throw std::invalid_argument("cannot import a null Metal device");
        // A framework may obtain its device through a different enumeration
        // route. Registry identity, rather than Objective-C pointer identity,
        // keeps one runtime backend for the actual native device.
        static std::mutex mutex;
        static std::unordered_map<uint64_t, std::weak_ptr<DeviceBackend>> devices;
        std::lock_guard<std::mutex> lock(mutex);
        auto& cached = devices[device.registryID];
        if (auto existing = cached.lock())
            return existing;
        auto created = std::make_shared<MetalDevice>(device);
        cached = created;
        return created;
    }
}
} // namespace detail
} // namespace metalrt
