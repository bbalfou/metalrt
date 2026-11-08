#include <metalrt/interop.h>
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <cmath>
#include <iostream>

namespace {
void check(bool condition, const char* message) {
    if (!condition)
        throw std::runtime_error(message);
}
template <class F> void rejects(F&& function, const char* message) {
    bool rejected = false;
    try {
        function();
    } catch (const std::exception&) {
        rejected = true;
    }
    check(rejected, message);
}
constexpr const char* shader = R"METAL(
#include <metal_stdlib>
using namespace metal;
kernel void interop_affine(device const float* input [[buffer(0)]],
                           device float* output [[buffer(1)]],
                           constant uint& count [[buffer(2)]],
                           uint i [[thread_position_in_grid]]) {
    if(i<count)output[i]=2.0f*input[i]+1.0f;
}
)METAL";

void native_sequence(metalrt::Device device, id<MTLDevice> native, metalrt::DispatchPolicy policy) {
    constexpr uint32_t count = 17;
    id<MTLCommandQueue> queue = [native newCommandQueue];
    id<MTLCommandBuffer> current = [queue commandBuffer];
    id<MTLBuffer> source = [native newBufferWithLength:128 options:MTLResourceStorageModeShared];
    id<MTLBuffer> private_input = [native newBufferWithLength:128
                                                      options:MTLResourceStorageModePrivate];
    id<MTLBuffer> output = [native newBufferWithLength:128 options:MTLResourceStorageModeShared];
    id<MTLBuffer> readback = [native newBufferWithLength:count * sizeof(float)
                                                 options:MTLResourceStorageModeShared];
    check(queue && current && source && private_input && output && readback,
          "native test allocations failed");
    for (uint32_t i = 0; i < 32; ++i)
        static_cast<float*>(source.contents)[i] = float(i) * 0.25f;

    auto owner = std::make_shared<int>(42);
    std::weak_ptr<int> weak_owner = owner;
    auto imported_input = metalrt::import_metal_buffer(device, (__bridge void*)private_input);
    auto imported_output = metalrt::import_metal_buffer(device, (__bridge void*)output, owner);
    auto duplicate = metalrt::import_metal_buffer(device, (__bridge void*)output, owner);
    check(duplicate == imported_output, "repeated imports must preserve allocation identity");
    check(imported_output.size() == output.length,
          "import must expose the full native buffer range");
    check(
        metalrt::resolve_pointer(static_cast<float*>(output.contents) + 7, count * sizeof(float)) ==
            imported_output.view(7 * sizeof(float), count * sizeof(float)),
        "Shared subviews must retain tensor byte offsets");
    rejects([&] { imported_input.contents(); }, "Private import must never expose a CPU pointer");
    rejects([&] { imported_output.view(100, 40); }, "native import bounds must be checked");

    // Caller work before and after the runtime is encoded into exactly the
    // same external command buffer, so both directions of ordering are tested.
    id<MTLBlitCommandEncoder> before = [current blitCommandEncoder];
    [before copyFromBuffer:source
              sourceOffset:5 * sizeof(float)
                  toBuffer:private_input
         destinationOffset:3 * sizeof(float)
                      size:count * sizeof(float)];
    [before endEncoding];
    auto stream = metalrt::borrow_metal_stream(
        device, [&]() -> void* { return (__bridge void*)current; }, policy);
    auto kernel = device.kernel(shader, "interop_affine");
    {
        auto batch = stream->batch();
        stream->dispatch(kernel, {count, 1, 1}, {32, 1, 1},
                         {metalrt::Argument::buffer(
                              0, imported_input.view(3 * sizeof(float), count * sizeof(float))),
                          metalrt::Argument::buffer(
                              1, imported_output.view(7 * sizeof(float), count * sizeof(float)),
                              metalrt::Access::Write),
                          metalrt::Argument::scalar(2, count)});
        batch.finish();
    }
    auto ticket = stream->last_ticket();
    check(current.status < MTLCommandBufferStatusCommitted,
          "runtime must not commit its owner's buffer");
    rejects([&] { ticket.wait(); },
            "waiting before external commit must reject instead of deadlocking");
    id<MTLBlitCommandEncoder> after = [current blitCommandEncoder];
    [after copyFromBuffer:output
             sourceOffset:7 * sizeof(float)
                 toBuffer:readback
        destinationOffset:0
                     size:count * sizeof(float)];
    [after endEncoding];

    owner.reset();
    duplicate = {};
    imported_output = {};
    imported_input = {};
    stream.reset(); // Must return before the caller can perform this commit.
    check(!weak_owner.expired(), "external allocator lease must survive release before commit");
    [current commit];
    ticket.wait();
    for (uint32_t i = 0; i < count; ++i)
        check(std::abs(static_cast<float*>(readback.contents)[i] -
                       (2.0f * float(i + 5) * 0.25f + 1.0f)) < 1e-6f,
              "caller/runtime/caller native ordering or storage offset is incorrect");

    auto invalid =
        metalrt::borrow_metal_stream(device, [&]() -> void* { return (__bridge void*)current; });
    auto scratch = device.allocate(4);
    rejects([&] { invalid->fill(scratch.view(), 0); },
            "committed external buffers must reject more encoding");
    auto absent = metalrt::borrow_metal_stream(device, []() -> void* { return nullptr; });
    rejects([&] { absent->fill(scratch.view(), 0); },
            "null provider result must fail before encoding");
}

void rotating_command_buffers(metalrt::Device device, id<MTLDevice> native) {
    id<MTLCommandQueue> queue = [native newCommandQueue];
    id<MTLCommandBuffer> current = [queue commandBuffer];
    auto stream =
        metalrt::borrow_metal_stream(device, [&]() -> void* { return (__bridge void*)current; });
    auto buffer = device.allocate(32);
    stream->fill(buffer.view(), 3);
    auto first = stream->last_ticket();
    [current commit];
    current = [queue commandBuffer];
    stream->fill(buffer.view(8, 8), 9);
    auto second = stream->last_ticket();
    [current commit];
    first.wait();
    second.wait();
    for (size_t i = 0; i < 32; ++i)
        check(buffer.data<uint8_t>()[i] == (i >= 8 && i < 16 ? 9 : 3),
              "persistent borrowed stream did not follow its owner's new buffer");
}

void explicit_untracked_sequence(metalrt::Device device, id<MTLDevice> native,
                                 metalrt::DispatchPolicy policy) {
    constexpr uint32_t count = 17;
    constexpr auto explicit_sync = metalrt::MetalBufferSynchronization::Explicit;
    id<MTLCommandQueue> queue = [native newCommandQueue];
    id<MTLCommandBuffer> current = [queue commandBufferWithUnretainedReferences];
    id<MTLEvent> boundary = [native newEvent];
    id<MTLBuffer> source = [native
        newBufferWithLength:128
                    options:MTLResourceStorageModeShared | MTLResourceHazardTrackingModeTracked];
    id<MTLBuffer> a = [native
        newBufferWithLength:128
                    options:MTLResourceStorageModeShared | MTLResourceHazardTrackingModeUntracked];
    id<MTLBuffer> b = [native
        newBufferWithLength:128
                    options:MTLResourceStorageModePrivate | MTLResourceHazardTrackingModeUntracked];
    id<MTLBuffer> c = [native
        newBufferWithLength:128
                    options:MTLResourceStorageModeShared | MTLResourceHazardTrackingModeUntracked];
    id<MTLBuffer> counts = [native
        newBufferWithLength:12
                    options:MTLResourceStorageModeShared | MTLResourceHazardTrackingModeUntracked];
    id<MTLBuffer> readback = [native
        newBufferWithLength:count * sizeof(float)
                    options:MTLResourceStorageModeShared | MTLResourceHazardTrackingModeTracked];
    check(queue && current && boundary && source && a && b && c && counts && readback,
          "untracked interop allocations failed");
    for (uint32_t i = 0; i < 32; ++i)
        static_cast<float*>(source.contents)[i] = float(i) * 0.25f;
    for (uint32_t i = 0; i < 3; ++i)
        static_cast<uint32_t*>(counts.contents)[i] = 1;
    auto owner = std::make_shared<int>(17);
    std::weak_ptr<int> weak_owner = owner;
    auto av = metalrt::import_metal_buffer(device, (__bridge void*)a, {}, explicit_sync);
    auto bv = metalrt::import_metal_buffer(device, (__bridge void*)b, {}, explicit_sync);
    auto cv = metalrt::import_metal_buffer(device, (__bridge void*)c, owner, explicit_sync);
    auto gv = metalrt::import_metal_buffer(device, (__bridge void*)counts, {}, explicit_sync);
    rejects([&] { metalrt::import_metal_buffer(device, (__bridge void*)c); },
            "cached explicit import must not silently opt a tracked-only caller into untracked "
            "resources");

    id<MTLBlitCommandEncoder> before = [current blitCommandEncoder];
    [before copyFromBuffer:source
              sourceOffset:5 * sizeof(float)
                  toBuffer:a
         destinationOffset:3 * sizeof(float)
                      size:count * sizeof(float)];
    [before endEncoding];
    [current encodeSignalEvent:boundary value:1];
    [current encodeWaitForEvent:boundary value:1];

    auto stream = metalrt::borrow_metal_stream(
        device, [&]() -> void* { return (__bridge void*)current; }, policy);
    auto kernel = device.kernel(shader, "interop_affine");
    {
        auto recording = stream->batch();
        stream->copy(av.view(3 * sizeof(float), count * sizeof(float)),
                     bv.view(2 * sizeof(float), count * sizeof(float)));
        stream->dispatch(
            kernel, {count, 1, 1}, {32, 1, 1},
            {metalrt::Argument::buffer(0, bv.view(2 * sizeof(float), count * sizeof(float))),
             metalrt::Argument::buffer(1, cv.view(7 * sizeof(float), count * sizeof(float)),
                                       metalrt::Access::Write),
             metalrt::Argument::scalar(2, count)});
        stream->copy(cv.view(7 * sizeof(float), count * sizeof(float)),
                     av.view(3 * sizeof(float), count * sizeof(float)));
        stream->dispatch_indirect(
            kernel, gv.view(), {1, 1, 1}, {32, 1, 1},
            {metalrt::Argument::buffer(0, av.view(3 * sizeof(float), count * sizeof(float))),
             metalrt::Argument::buffer(1, bv.view(2 * sizeof(float), count * sizeof(float)),
                                       metalrt::Access::Write),
             metalrt::Argument::scalar(2, count)});
        stream->copy(bv.view(2 * sizeof(float) + 1, 5), cv.view(100, 5));
        recording.finish();
    }
    // A distinct runtime submission must continue the same fence chain.
    stream->dispatch(
        kernel, {count, 1, 1}, {32, 1, 1},
        {metalrt::Argument::buffer(0, bv.view(2 * sizeof(float), count * sizeof(float))),
         metalrt::Argument::buffer(1, cv.view(7 * sizeof(float), count * sizeof(float)),
                                   metalrt::Access::Write),
         metalrt::Argument::scalar(2, count)});
    auto ticket = stream->last_ticket();
    check(current.status < MTLCommandBufferStatusCommitted,
          "untracked runtime work must not commit MLX-style buffers");
    [current encodeSignalEvent:boundary value:2];
    [current encodeWaitForEvent:boundary value:2];
    id<MTLBlitCommandEncoder> after = [current blitCommandEncoder];
    [after copyFromBuffer:c
             sourceOffset:7 * sizeof(float)
                 toBuffer:readback
        destinationOffset:0
                     size:count * sizeof(float)];
    [after endEncoding];
    owner.reset();
    av = {};
    bv = {};
    cv = {};
    gv = {};
    kernel = {};
    stream.reset();
    check(!weak_owner.expired(),
          "untracked imported storage lease must survive borrowed stream destruction");
    [current commit];
    ticket.wait();
    for (uint32_t i = 0; i < count; ++i)
        check(std::abs(static_cast<float*>(readback.contents)[i] -
                       (8.0f * float(i + 5) * 0.25f + 7.0f)) < 1e-6f,
              "explicit untracked producer/runtime/consumer fence chain failed");
}

void rotating_untracked_buffers(metalrt::Device device, id<MTLDevice> native) {
    id<MTLCommandQueue> queue = [native newCommandQueue];
    id<MTLCommandBuffer> current = [queue commandBufferWithUnretainedReferences];
    id<MTLBuffer> native_buffer = [native
        newBufferWithLength:32
                    options:MTLResourceStorageModeShared | MTLResourceHazardTrackingModeUntracked];
    auto imported = metalrt::import_metal_buffer(device, (__bridge void*)native_buffer, {},
                                                 metalrt::MetalBufferSynchronization::Explicit);
    auto stream =
        metalrt::borrow_metal_stream(device, [&]() -> void* { return (__bridge void*)current; });
    auto readback = device.allocate(32);
    stream->fill(imported.view(), 3);
    [current commit];
    current = [queue commandBufferWithUnretainedReferences];
    stream->fill(imported.view(8, 8), 9);
    stream->copy(imported.view(), readback.view());
    auto ticket = stream->last_ticket();
    [current commit];
    ticket.wait();
    for (size_t i = 0; i < 32; ++i)
        check(readback.data<uint8_t>()[i] == (i >= 8 && i < 16 ? 9 : 3),
              "untracked fence chain must persist across command buffers");
    id<MTLCommandQueue> other_queue = [native newCommandQueue];
    current = [other_queue commandBuffer];
    rejects([&] { stream->fill(imported.view(), 0); },
            "untracked borrowed stream must reject a change of native queue");
}

void commit_dependency_retry(metalrt::Device device, id<MTLDevice> native) {
    id<MTLCommandQueue> producer_queue = [native newCommandQueue];
    id<MTLCommandQueue> consumer_queue = [native newCommandQueue];
    id<MTLCommandBuffer> producer_buffer = [producer_queue commandBuffer];
    id<MTLCommandBuffer> consumer_buffer = [consumer_queue commandBuffer];
    auto producer = metalrt::borrow_metal_stream(
        device, [&]() -> void* { return (__bridge void*)producer_buffer; });
    auto consumer = metalrt::borrow_metal_stream(
        device, [&]() -> void* { return (__bridge void*)consumer_buffer; });
    auto buffer = device.allocate(16);
    producer->fill(buffer.view(), 5);
    metalrt::TimelineEvent event{device};
    auto point = producer->signal(event);
    consumer->wait(point);
    consumer->fill(buffer.view(4, 4), 8);
    [consumer_buffer commit];
    rejects([&] { consumer->synchronize(); },
            "consumer must reject an uncommitted producer before waiting on its GPU event");
    [producer_buffer commit];
    consumer->synchronize();
    producer->synchronize();
    for (size_t i = 0; i < 16; ++i)
        check(buffer.data<uint8_t>()[i] == (i >= 4 && i < 8 ? 8 : 5),
              "cross-queue wait must survive retry after producer commit");
}
} // namespace

int main() {
    @autoreleasepool {
        id<MTLDevice> native = MTLCreateSystemDefaultDevice();
        if (!native) {
            std::cout << "SKIP: no Metal device\n";
            return 77;
        }
        try {
            auto device = metalrt::import_metal_device((__bridge void*)native);
            check(device == metalrt::import_metal_device((__bridge void*)native),
                  "device import must be canonical");
            check(device == metalrt::Device::system_default(),
                  "system_default must share the imported native identity");
            check((__bridge id<MTLDevice>)metalrt::native_metal_device(device) == native,
                  "native device export mismatch");
            id<MTLBuffer> untracked =
                [native newBufferWithLength:16
                                    options:MTLResourceStorageModeShared |
                                            MTLResourceHazardTrackingModeUntracked];
            check(untracked != nil, "untracked test allocation failed");
            rejects([&] { metalrt::import_metal_buffer(device, (__bridge void*)untracked); },
                    "untracked external resources must not silently bypass a fence contract");
            native_sequence(device, native, metalrt::DispatchPolicy::Direct);
            native_sequence(device, native, metalrt::DispatchPolicy::PerKernel);
            explicit_untracked_sequence(device, native, metalrt::DispatchPolicy::Direct);
            explicit_untracked_sequence(device, native, metalrt::DispatchPolicy::PerKernel);
            if (device.capabilities().indirect_compute)
                native_sequence(device, native, metalrt::DispatchPolicy::Indirect);
            if (device.capabilities().indirect_compute)
                explicit_untracked_sequence(device, native, metalrt::DispatchPolicy::Indirect);
            rotating_command_buffers(device, native);
            rotating_untracked_buffers(device, native);
            commit_dependency_retry(device, native);
            std::cout << "native Metal import/offset/lease/borrowed-queue checks passed\n";
            return 0;
        } catch (const std::exception& error) {
            std::cerr << error.what() << '\n';
            return 1;
        }
    }
}
