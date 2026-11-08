// Validation contracts across supported backends and dispatch policies.
#include "support.h"
namespace runtime_tests {
void pointers_and_validation(Device d, const Kernels& k) {
    auto memory = d.allocate(64);
    auto interior = resolve_pointer(memory.data<float>() + 3, 2 * sizeof(float));
    check(interior.buffer == memory && interior.offset == 12 && interior.length == 8,
          "interior pointer resolution");
    check(resolve_pointer(static_cast<uint8_t*>(memory.contents()) + 64, 0).length == 0,
          "empty one-past-end view");
    throws([&] { resolve_pointer(memory.data<float>() + 15, 8); }, "overrun must be rejected");
    throws([&] { memory.view(65, 0); }, "out-of-range offset must be rejected");
    throws([&] { memory.view(4, 61); }, "out-of-range length must be rejected");
    float ordinary = 0;
    throws([&] { resolve_pointer(&ordinary, 4); },
           "ordinary host memory must not become a GPU pointer");
    void* retired = nullptr;
    {
        auto temporary = d.allocate(16);
        retired = temporary.contents();
    }
    throws([&] { resolve_pointer(retired, 4); }, "retired allocation must not resolve");
    Stream stream(d);
    throws([&] { stream.copy(memory.view(0, 16), memory.view(4, 16)); },
           "overlapping copy must be rejected");
    throws([&] { stream.copy(memory.view(0, 16), memory.view(32, 12)); },
           "mismatched copy lengths must be rejected");
    throws(
        [&] {
            stream.dispatch(
                k.affine, {1, 1, 1}, {1, 1, 1},
                {Argument::buffer(0, memory.view()), Argument::buffer(0, memory.view())});
        },
        "duplicate buffer slot must be rejected");
    throws([&] { stream.dispatch(k.affine, {1, 1, 1}, {1025, 1, 1}, {}); },
           "oversized group must be rejected");
    Device other{detail::make_reference_device()};
    auto alien = other.allocate(64);
    throws([&] { affine(stream, k.affine, alien.view(), memory.view(), 1, 1, 0); },
           "cross-device buffer must be rejected");
    stream.copy(memory.view(0, 16), memory.view(0, 16));
    check(stream.statistics().command_buffers == 0, "self-copy should be a no-op");
    stream.dispatch(k.affine, {0, 1, 1}, {64, 1, 1}, {});
    check(stream.statistics().command_buffers == 0, "empty dispatch should be a no-op");
    throws([&] { checked_mul(SIZE_MAX, 2); }, "size multiplication overflow");
    throws([&] { checked_add(SIZE_MAX, 1); }, "size addition overflow");
}

void empty_recapture_releases_plan(Device d, const Kernels& k) {
    if (!d.capabilities().indirect_compute)
        return;
    Stream stream(d, DispatchPolicy::Indirect);
    auto input = d.allocate(4), output = d.allocate(4);
    *input.data<float>() = 2;
    std::weak_ptr<detail::BufferAllocation> input_weak = input.implementation();
    Graph graph;
    {
        auto g = graph.capture_guard(&stream);
        affine(stream, k.affine, input.view(), output.view(), 1, 2, 0);
    }
    stream.synchronize();
    {
        auto g = graph.capture_guard(&stream);
    }
    input = {};
    check(graph.size() == 0, "empty capture should replace the graph with an empty sequence");
    // Native completion tickets may retain Metal completion handlers beyond GPU
    // completion. The deterministic reference backend has no such extra retention.
    if (d.capabilities().backend == BackendKind::Reference)
        check(input_weak.expired(), "empty re-capture must release the previous cached plan");
}

} // namespace runtime_tests
