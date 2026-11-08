// Recording contracts across supported backends and dispatch policies.
#include "support.h"
namespace runtime_tests {
void cancellation_and_nested_capture(Device d, const Kernels& k) {
    Stream stream(d);
    auto input = d.allocate(4), temp = d.allocate(4), output = d.allocate(4);
    *input.data<float>() = 3;
    *output.data<float>() = -7;
    Graph graph, nested;
    try {
        auto guard = graph.capture_guard(&stream);
        affine(stream, k.affine, input.view(), output.view(), 1, 9, 0);
        throw std::runtime_error("intentional stack unwinding");
    } catch (const std::runtime_error&) {
    }
    check(graph.size() == 0 && stream.statistics().command_buffers == 0,
          "unwinding must discard pending commands");
    check(!stream.is_recording(), "unwinding must restore stream state");
    near(*output.data<float>(), -7, "cancelled capture must not write output");
    {
        auto outer = graph.capture_guard(&stream);
        auto inner = nested.capture_guard(&stream);
        check(!static_cast<bool>(inner), "nested capture must join its parent");
        affine(stream, k.affine, input.view(), temp.view(), 1, 2, 1);
        square(stream, k.square, temp.view(), output.view(), 1);
    }
    stream.synchronize();
    check(graph.size() == 2 && nested.size() == 0,
          "nested capture commands must belong to the outer graph");
    near(*output.data<float>(), 49, "nested dependent result");
    {
        auto guard = graph.capture_guard(&stream);
        affine(stream, k.affine, input.view(), output.view(), 1, 100, 0);
        guard.cancel();
    }
    check(graph.size() == 2, "cancelled re-capture must preserve the last successful graph");
    graph.launch(&stream).wait();
    stream.synchronize();
    near(*output.data<float>(), 49, "replay after cancellation");
    {
        auto batch = stream.batch();
        auto nested_batch = stream.batch();
        check(!static_cast<bool>(nested_batch), "nested batch should be a no-op scope");
        affine(stream, k.affine, input.view(), output.view(), 1, 3, 2);
    }
    stream.synchronize();
    near(*output.data<float>(), 11, "batch destructor must submit");
    {
        auto guard = graph.capture_guard(&stream);
        throws([&] { graph.reset(); }, "cannot reset an active capture");
        throws([&] { stream.synchronize(); }, "cannot wait while capturing");
        guard.cancel();
    }
    graph.schedule_synchronize();
    {
        auto guard = graph.capture_guard(&stream);
        affine(stream, k.affine, input.view(), output.view(), 1, 2, 0);
    }
    near(*output.data<float>(), 6,
         "scheduled capture synchronization must complete before scope exit");
}

void cpu_boundaries(Device d, const Kernels& k) {
    Stream stream(d);
    auto input = d.allocate(4), intermediate = d.allocate(4), output = d.allocate(4);
    *input.data<float>() = 2;
    int callbacks = 0;
    {
        auto batch = stream.batch();
        affine(stream, k.affine, input.view(), intermediate.view(), 1, 3, 1);
        stream.host([&] {
            ++callbacks;
            near(*intermediate.data<float>(), 7, "CPU reads completed GPU value");
            *input.data<float>() = 8;
        });
        square(stream, k.square, input.view(), output.view(), 1);
    }
    stream.synchronize();
    check(callbacks == 1, "host boundary executes exactly once");
    check(stream.statistics().command_buffers == 2, "CPU boundary splits direct batch");
    near(*output.data<float>(), 64, "GPU work after host boundary sees CPU write");
    Graph graph;
    auto capture = graph.capture_guard(&stream);
    throws([&] { stream.host([&] { ++callbacks; }); },
           "host callback must be rejected during reusable capture");
    capture.cancel();
    check(callbacks == 1, "rejected host callback must not run");
}

void capture_thread_rules(Device d, const Kernels& k) {
    Stream stream(d), other_stream(d);
    Graph graph;
    auto input = d.allocate(4), output = d.allocate(4);
    *input.data<float>() = 1;
    auto guard = graph.capture_guard(&stream);
    bool rejected = false;
    std::thread wrong_thread([&] {
        try {
            affine(stream, k.affine, input.view(), output.view(), 1, 1, 0);
        } catch (const std::exception&) {
            rejected = true;
        }
    });
    wrong_thread.join();
    check(rejected, "another host thread must not add commands to an active capture");
    throws([&] { auto second = graph.capture_guard(&other_stream); },
           "one graph cannot capture two streams concurrently");
    guard.cancel();
}

void default_stream_capture(Device d, const Kernels& k) {
    Device::set_current(d);
    auto input = d.allocate(4), output = d.allocate(4);
    *input.data<float>() = 4;
    Graph graph;
    {
        auto guard = graph.capture_guard();
        check(default_stream().is_capturing(), "null-stream capture uses explicit default stream");
        affine(default_stream(), k.affine, input.view(), output.view(), 1, 3, 2);
    }
    default_stream().synchronize();
    near(*output.data<float>(), 14, "default-stream result");
}

} // namespace runtime_tests
