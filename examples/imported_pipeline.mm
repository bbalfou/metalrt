#include <metalrt/interop.h>
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <iostream>

// An external execution owner can create its own library and pipelines and hand
// retained handles to metalrt. No shader compilation occurs during import.
int main() {
    @autoreleasepool {
        try {
            auto device = metalrt::Device::system_default();
            auto native = (__bridge id<MTLDevice>)metalrt::native_metal_device(device);
            NSError* error = nil;
            auto library = [native
                newLibraryWithSource:@"#include <metal_stdlib>\n"
                                      "using namespace metal;\nkernel void native_value(device "
                                      "uint* p [[buffer(0)]]) { p[0]=42; }"
                             options:nil
                               error:&error];
            if (!library)
                throw std::runtime_error("external library creation failed");
            auto function = [library newFunctionWithName:@"native_value"];
            auto pipeline = [native newComputePipelineStateWithFunction:function error:&error];
            if (!pipeline)
                throw std::runtime_error("external pipeline creation failed");
            auto kernel =
                metalrt::import_metal_pipeline(device, (__bridge void*)pipeline, "native_value");
            pipeline = nil;
            library = nil;
            function = nil; // Runtime kernel retains the pipeline.
            metalrt::Stream stream{device};
            auto output = device.allocate(sizeof(uint32_t));
            stream.dispatch(kernel, {1, 1, 1}, {1, 1, 1},
                            {metalrt::Argument::buffer(0, output.view(), metalrt::Access::Write)});
            stream.synchronize();
            if (output.data<uint32_t>()[0] != 42)
                throw std::runtime_error("imported result mismatch");
            std::cout << "Imported pipeline returned " << output.data<uint32_t>()[0] << '\n';
        } catch (const std::exception& e) {
            std::cerr << e.what() << '\n';
            return 1;
        }
    }
}
