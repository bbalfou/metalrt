#include <metalrt/interop.h>
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <iostream>
using namespace metalrt;
namespace {
void check(bool value, const char* text) {
    if (!value)
        throw std::runtime_error(text);
}
template <class F> void rejects(F f) {
    bool rejected = false;
    try {
        f();
    } catch (const std::exception&) {
        rejected = true;
    }
    check(rejected, "invalid pipeline import/use accepted");
}
} // namespace
int main() {
    @autoreleasepool {
        try {
            Device device;
            try {
                device = Device::system_default();
            } catch (const std::exception& e) {
                std::cout << "SKIP: " << e.what() << '\n';
                return 77;
            }
            auto native = (__bridge id<MTLDevice>)native_metal_device(device);
            NSError* error = nil;
            auto library = [native
                newLibraryWithSource:
                    @"#include <metal_stdlib>\nusing namespace metal;\n"
                     "kernel void imported_value(device uint* out [[buffer(0)]]) { out[0]=73; }"
                             options:nil
                               error:&error];
            check(library != nil, "native library creation failed");
            auto function = [library newFunctionWithName:@"imported_value"];
            auto descriptor = [MTLComputePipelineDescriptor new];
            descriptor.computeFunction = function;
            auto direct = [native newComputePipelineStateWithDescriptor:descriptor
                                                                options:0
                                                             reflection:nil
                                                                  error:&error];
            check(direct != nil, "native direct pipeline creation failed");
            const auto before = device.kernel_cache_statistics();
            rejects([&] { import_metal_pipeline(device, nullptr, "missing"); });
            rejects([&] { import_metal_pipeline(device, (__bridge void*)direct, ""); });
            rejects([&] {
                import_metal_pipeline(Device::reference(), (__bridge void*)direct, "invalid");
            });
            rejects([&] {
                import_metal_pipeline(device, (__bridge void*)direct, "bad",
                                      (__bridge void*)direct);
            });
            auto only_direct =
                import_metal_pipeline(device, (__bridge void*)direct, "imported_value");
            Kernel both;
            if (device.capabilities().indirect_compute) {
                descriptor.supportIndirectCommandBuffers = YES;
                auto indirect = [native newComputePipelineStateWithDescriptor:descriptor
                                                                      options:0
                                                                   reflection:nil
                                                                        error:&error];
                check(indirect != nil, "native ICB pipeline creation failed");
                both = import_metal_pipeline(device, (__bridge void*)direct, "imported_value",
                                             (__bridge void*)indirect);
                indirect = nil;
            }
            for (id<MTLDevice> other in MTLCopyAllDevices()) {
                if (other.registryID == native.registryID)
                    continue;
                rejects([&] {
                    import_metal_pipeline(import_metal_device((__bridge void*)other),
                                          (__bridge void*)direct, "wrong_device");
                });
            }
            direct = nil;
            function = nil;
            library = nil;
            descriptor = nil;
            check(device.kernel_cache_statistics().source_compilations ==
                      before.source_compilations,
                  "pipeline import compiled source");
            for (auto policy :
                 {DispatchPolicy::Direct, DispatchPolicy::PerKernel, DispatchPolicy::Indirect}) {
                if (policy == DispatchPolicy::Indirect && !both)
                    continue;
                Stream stream{device, policy};
                auto output = device.allocate(sizeof(uint32_t));
                Graph graph;
                {
                    auto record = graph.record_guard(&stream);
                    stream.dispatch(policy == DispatchPolicy::Indirect ? both : only_direct,
                                    {1, 1, 1}, {1, 1, 1},
                                    {Argument::buffer(0, output.view(), Access::Write)});
                    record.finish();
                }
                device.clear_kernel_cache();
                graph.launch(&stream).wait();
                check(output.data<uint32_t>()[0] == 73, "imported pipeline execution failed");
                if (policy == DispatchPolicy::Indirect)
                    rejects([&] {
                        stream.dispatch(only_direct, {1, 1, 1}, {1, 1, 1},
                                        {Argument::buffer(0, output.view(), Access::Write)});
                    });
            }
            if (device.capabilities().kernel_timestamps) {
                auto queue = [native newCommandQueue];
                auto current = [queue commandBuffer];
                auto borrowed =
                    borrow_metal_stream(device, [&] { return (__bridge void*)current; });
                auto output = device.allocate(sizeof(uint32_t));
                GpuTimingOptions timing;
                timing.sample_kernels = true;
                borrowed->begin_gpu_timing(timing);
                borrowed->dispatch(only_direct, {1, 1, 1}, {1, 1, 1},
                                   {Argument::buffer(0, output.view(), Access::Write)});
                rejects([&] { borrowed->last_ticket().wait(); });
                [current commit];
                const auto samples = borrowed->end_gpu_timing();
                check(samples.kernels.size() == 1 && samples.kernels.front().valid,
                      "borrowed command buffer lost counter samples");
                check(output.data<uint32_t>()[0] == 73, "sampled imported pipeline failed");
            }
            std::cout << "PASS: native pipeline import, ownership, graph replay and validation\n";
        } catch (const std::exception& e) {
            std::cerr << e.what() << '\n';
            return 1;
        }
    }
}
