#include <metalrt/runtime.h>
#include <metalrt/detail/backend.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <future>
#include <iostream>
#include <mutex>
#include <string>

#ifndef METALRT_TEST_SHADER_DIR
#define METALRT_TEST_SHADER_DIR "kernels"
#endif

using namespace metalrt;
namespace {
struct Args {
    uint32_t count;
    float scale, bias;
};
static_assert(sizeof(Args) == 12, "graph_features.metal constant ABI");
const std::string shader = std::string{METALRT_TEST_SHADER_DIR} + "/graph_features.metal";
void check(bool value, const std::string& message) {
    if (!value)
        throw std::runtime_error(message);
}
void near(float actual, float expected, const char* message) {
    check(std::isfinite(actual) &&
              std::abs(actual - expected) <= 1e-5f * std::max(1.0f, std::abs(expected)),
          message);
}
template <class F> void throws(F&& action, const char* message) {
    try {
        action();
    } catch (const std::exception&) {
        return;
    }
    throw std::runtime_error(message);
}
CpuKernel affine_reference() {
    return [](const ComputeCommand& command) {
        const auto args = command.scalar<Args>(2);
        const auto* input = command.buffer<const float>(0);
        auto* output = command.buffer<float>(1);
        for (size_t i = 0; i < std::min<size_t>(args.count, command.grid.x); ++i)
            output[i] = input[i] * args.scale + args.bias;
    };
}
struct Kernels {
    Kernel affine, sum;
    explicit Kernels(Device device)
        : affine(device.kernel_file(shader, "graph_affine", affine_reference())),
          sum(device.kernel_file(shader, "graph_sum", [](const ComputeCommand& command) {
              const auto count = command.scalar<uint32_t>(3);
              const auto* a = command.buffer<const float>(0);
              const auto* b = command.buffer<const float>(1);
              auto* output = command.buffer<float>(2);
              for (size_t i = 0; i < std::min<size_t>(count, command.grid.x); ++i)
                  output[i] = a[i] + b[i];
          })) {}
};
void affine(Stream& stream, const Kernel& kernel, BufferView input, BufferView output, Args args) {
    launch_linear(&stream, kernel, args.count,
                  {Argument::buffer(0, std::move(input)),
                   Argument::buffer(1, std::move(output), Access::Write),
                   Argument::scalar(2, args)},
                  64);
}
void record(Graph& graph, Stream& stream, const Kernel& kernel, BufferView input, BufferView output,
            Args args) {
    auto scope = graph.record_guard(&stream);
    affine(stream, kernel, std::move(input), std::move(output), args);
    scope.finish();
}
void execute(Graph& graph, Stream& stream) {
    graph.launch(&stream).wait();
    stream.synchronize();
}
const char* policy_name(DispatchPolicy policy) {
    return policy == DispatchPolicy::PerKernel ? "per-kernel"
           : policy == DispatchPolicy::Direct  ? "direct"
                                               : "indirect";
}

void recording_and_preparation(Device device, const Kernels& kernels, DispatchPolicy policy) {
    constexpr uint32_t count = 37;
    Stream stream{device, policy};
    auto input = device.allocate(count * sizeof(float)),
         output = device.allocate(count * sizeof(float));
    for (uint32_t i = 0; i < count; ++i) {
        input.data<float>()[i] = float(i);
        output.data<float>()[i] = -123;
    }
    Graph graph;
    record(graph, stream, kernels.affine, input.view(), output.view(), {count, 2, 1});
    check(graph.size() == 1 && graph.revision() == 1,
          "recording should publish one command/revision");
    check(stream.statistics().command_buffers == 0, "record_guard must not execute");
    graph.prepare(&stream);
    auto stats = stream.statistics();
    check(stats.prepare_calls == 1, "preparation must be tracked");
    check(stats.command_buffers == 0 && stats.compute_dispatches == 0,
          "preparation must not dispatch GPU work");
    for (uint32_t i = 0; i < count; ++i)
        near(output.data<float>()[i], -123, "preparation changed output");
    execute(graph, stream);
    execute(graph, stream);
    for (uint32_t i = 0; i < count; ++i)
        near(output.data<float>()[i], 2 * float(i) + 1, "prepared graph result");
    stats = stream.statistics();
    if (policy == DispatchPolicy::Indirect)
        check(stats.icb_builds == 1 && stats.icb_reuses == 2 && stats.icb_updates == 0,
              "unchanged prepared graph must replay its ICB");
}

void update_bindings_scalars_grid(Device device, const Kernels& kernels, DispatchPolicy policy) {
    constexpr uint32_t count = 16;
    Stream stream{device, policy};
    auto a = device.allocate((count + 8) * sizeof(float)),
         b = device.allocate((count + 8) * sizeof(float));
    auto replacement = device.allocate((count + 8) * sizeof(float));
    auto output = device.allocate(count * sizeof(float)),
         shifted = device.allocate((count + 6) * sizeof(float));
    for (uint32_t i = 0; i < count + 8; ++i) {
        a.data<float>()[i] = float(i);
        b.data<float>()[i] = float(i) + 20;
        replacement.data<float>()[i] = float(i) + 100;
    }
    for (uint32_t i = 0; i < count + 6; ++i)
        shifted.data<float>()[i] = -1000;
    const auto input_view = a.view(4 * sizeof(float), count * sizeof(float));
    Graph graph;
    record(graph, stream, kernels.affine, input_view, output.view(), {count, 2, 1});
    graph.prepare(&stream);
    execute(graph, stream);
    const auto old_snapshot = graph.commands();
    const auto revision = graph.revision();
    auto edit = graph.update();
    edit.set_buffer(0, 0, b.view(4 * sizeof(float), count * sizeof(float)))
        .set_scalar(0, 2, Args{count, 0.5f, -1});
    edit.commit();
    check(graph.revision() == revision + 1, "one transaction should publish one revision");
    check(old_snapshot[0].compute.argument(0).resource == input_view &&
              old_snapshot[0].compute.scalar<Args>(2).scale == 2,
          "inspection snapshots must retain the old bindings/constants");
    execute(graph, stream);
    for (uint32_t i = 0; i < count; ++i)
        near(output.data<float>()[i], 0.5f * (float(i) + 24) - 1, "scalar/input update result");

    auto smaller = graph.update();
    smaller.set_buffer(0, 1, shifted.view(3 * sizeof(float), count * sizeof(float)))
        .set_grid(0, {8, 1, 1}, {1, 1, 1});
    smaller.commit();
    execute(graph, stream);
    for (uint32_t i = 0; i < count + 6; ++i)
        near(shifted.data<float>()[i], i >= 3 && i < 11 ? 0.5f * (float(i - 3) + 24) - 1 : -1000,
             "grid/subview update or untouched guard");
    auto rebinding = graph.update();
    rebinding.rebind(b, replacement);
    rebinding.commit();
    execute(graph, stream);
    check(graph.commands()[0].compute.argument(0).resource.offset == 4 * sizeof(float),
          "whole-allocation rebind lost a view offset");
    for (uint32_t i = 0; i < 8; ++i)
        near(shifted.data<float>()[i + 3], 0.5f * (float(i) + 104) - 1,
             "whole-allocation rebind result");
    if (policy == DispatchPolicy::Indirect)
        check(stream.statistics().icb_updates >= 1,
              "compatible idle graph edit should update its plan");
}

void independent_clones(Device device, const Kernels& kernels, DispatchPolicy policy) {
    Stream stream{device, policy};
    auto a = device.allocate(sizeof(float)), b = device.allocate(sizeof(float));
    auto first = device.allocate(sizeof(float)), second = device.allocate(sizeof(float));
    a.data<float>()[0] = 3;
    b.data<float>()[0] = 9;
    Graph graph;
    record(graph, stream, kernels.affine, a.view(), first.view(), {1, 2, 1});
    graph.prepare(&stream);
    auto clone = graph.clone();
    auto edit = clone.update();
    edit.set_buffer(0, 0, b.view())
        .set_buffer(0, 1, second.view())
        .set_scalar(0, 2, Args{1, 3, -2});
    edit.commit();
    execute(clone, stream);
    execute(graph, stream);
    const auto warmed = stream.statistics();
    execute(clone, stream);
    execute(graph, stream);
    if (policy == DispatchPolicy::Indirect)
        check(stream.statistics().icb_builds == warmed.icb_builds &&
                  stream.statistics().icb_updates == warmed.icb_updates,
              "alternating prepared clones re-encoded each other's mutable plan cache");
    near(first.data<float>()[0], 7, "original graph changed with its clone");
    near(second.data<float>()[0], 25, "clone edit was not independent");
    clone.reset();
    check(graph.size() == 1, "resetting a clone reset the original");
}

void transactional_validation(Device device, const Kernels& kernels) {
    Stream stream{device};
    auto input = device.allocate(16), output = device.allocate(16);
    Graph graph;
    record(graph, stream, kernels.affine, input.view(), output.view(), {4, 1, 0});
    auto first = graph.update(), stale = graph.update();
    first.set_scalar(0, 2, Args{4, 2, 1}).commit();
    throws([&] { stale.set_scalar(0, 2, Args{4, 9, 9}).commit(); },
           "conflicting transaction was accepted");
    const auto revision = graph.revision();
    const auto snapshot = graph.commands();
    throws([&] { graph.update().set_scalar(0, 2, uint32_t{4}).commit(); },
           "constant ABI change was accepted");
    throws([&] { graph.update().set_scalar(0, 0, Args{4, 1, 0}).commit(); },
           "buffer argument changed to a scalar");
    throws([&] { graph.update().set_buffer(99, 0, input.view()).commit(); },
           "invalid command index was accepted");
    throws([&] { graph.update().set_grid(0, {0, 1, 1}, {1, 1, 1}).commit(); },
           "zero grid was accepted by an update");
    auto too_small = device.allocate(4);
    throws([&] { graph.update().rebind(input, too_small).commit(); },
           "undersized whole-buffer rebind was accepted");
    Device other{detail::make_reference_device()};
    auto alien = other.allocate(16);
    throws([&] { graph.update().set_buffer(0, 0, alien.view()).commit(); },
           "cross-device update was accepted");
    check(graph.revision() == revision && detail::same_commands(graph.commands(), snapshot),
          "failed transaction changed the published graph");
    auto recapture_conflict = graph.update();
    record(graph, stream, kernels.affine, input.view(), output.view(), {4, 3, 2});
    throws([&] { recapture_conflict.commit(); },
           "recapture did not invalidate a stale transaction");
    auto reset_conflict = graph.update();
    graph.reset();
    throws([&] { reset_conflict.commit(); }, "reset did not invalidate a stale transaction");
    check(graph.size() == 0, "conflicting transaction resurrected a reset graph");
}

void diamond_parallel(Device device, const Kernels& kernels, DispatchPolicy policy) {
    constexpr uint32_t count = 37;
    const size_t bytes = count * sizeof(float);
    Stream stream{device, policy};
    auto input = device.allocate(bytes), x = device.allocate(bytes),
         branches = device.allocate(2 * bytes), output = device.allocate(bytes);
    for (uint32_t i = 0; i < count; ++i)
        input.data<float>()[i] = float(i) / 4;
    const auto y = branches.view(0, bytes), z = branches.view(bytes, bytes);
    Graph graph;
    {
        auto recording = graph.record_guard(&stream);
        affine(stream, kernels.affine, input.view(), x.view(), {count, 2, 1});
        {
            auto parallel = stream.parallel();
            affine(stream, kernels.affine, x.view(), y, {count, 3, 0});
            affine(stream, kernels.affine, x.view(), z, {count, -1, 4});
            parallel.finish();
        }
        launch_linear(&stream, kernels.sum, count,
                      {Argument::buffer(0, y), Argument::buffer(1, z),
                       Argument::buffer(2, output.view(), Access::Write),
                       Argument::scalar(3, count)});
        recording.finish();
    }
    const auto snapshot = graph.commands();
    check(snapshot.size() == 4 && snapshot[1].compute.parallel_group != 0 &&
              snapshot[1].compute.parallel_group == snapshot[2].compute.parallel_group,
          "parallel group identity was not retained");
    check(snapshot[1].compute.barrier_before && !snapshot[2].compute.barrier_before &&
              snapshot[3].compute.barrier_before,
          "parallel group entry/join barriers are wrong");
    graph.prepare(&stream);
    execute(graph, stream);
    execute(graph, stream);
    for (uint32_t i = 0; i < count; ++i)
        near(output.data<float>()[i], 4 * input.data<float>()[i] + 6, "parallel diamond result");
    const auto revision = graph.revision();
    throws([&] { graph.update().set_buffer(1, 1, z).commit(); },
           "graph update introduced overlapping parallel writers");
    check(graph.revision() == revision && detail::same_commands(graph.commands(), snapshot),
          "failed parallel update mutated the graph");
}

void parallel_validation(Device device, const Kernels& kernels) {
    Stream stream{device};
    auto input = device.allocate(64), shared = device.allocate(64), output = device.allocate(64);
    Graph graph;
    auto recording = graph.record_guard(&stream);
    auto parallel = stream.parallel();
    affine(stream, kernels.affine, input.view(), shared.view(), {16, 1, 0});
    throws([&] { affine(stream, kernels.affine, shared.view(), output.view(), {16, 1, 0}); },
           "parallel read-after-write was accepted");
    throws([&] { affine(stream, kernels.affine, input.view(), shared.view(4, 32), {8, 1, 0}); },
           "partially overlapping parallel writers were accepted");
    throws([&] { auto nested = stream.parallel(); }, "nested parallel group was accepted");
    throws([&] { stream.fill(output.view()); },
           "transfer inside compute-only parallel group was accepted");
    parallel.cancel();
    recording.cancel();
    check(!stream.is_recording() && graph.size() == 0 && stream.statistics().command_buffers == 0,
          "cancelled parallel group leaked recording or work");

    // Validation failures must not consume a parallel group's first-command
    // barrier or the join barrier required by the next accepted command.
    {
        auto scope = graph.record_guard(&stream);
        affine(stream, kernels.affine, input.view(), shared.view(), {16, 1, 0});
        {
            auto group = stream.parallel();
            throws(
                [&] {
                    stream.dispatch(
                        kernels.affine, {16, 1, 1}, {64, 1, 1},
                        {Argument::buffer(0, shared.view()), Argument::buffer(0, input.view())});
                },
                "duplicate argument inside parallel group was accepted");
            affine(stream, kernels.affine, shared.view(), output.view(), {16, 1, 0});
            group.finish();
        }
        Command join;
        join.compute.kernel = kernels.affine;
        join.compute.grid = {16, 1, 1};
        join.compute.group = {64, 1, 1};
        join.compute.barrier_before = false;
        join.compute.arguments = {Argument::buffer(0, output.view()),
                                  Argument::buffer(1, input.view(), Access::Write),
                                  Argument::scalar(2, Args{16, 1, 0})};
        Command invalid = join;
        invalid.compute.arguments.push_back(Argument::buffer(0, shared.view()));
        throws([&] { stream.enqueue(invalid); },
               "duplicate argument after parallel group was accepted");
        stream.enqueue(join);
        scope.finish();
    }
    const auto commands = graph.commands();
    check(commands.size() == 3 && commands[1].compute.barrier_before &&
              commands[2].compute.barrier_before,
          "invalid command consumed the next accepted command's dependency barrier");
}

void cross_stream_events(Device device, const Kernels& kernels, DispatchPolicy policy) {
    constexpr uint32_t count = 17;
    Stream producer{device, policy}, consumer{device, policy};
    TimelineEvent event{device};
    auto input = device.allocate(count * sizeof(float)),
         shared = device.allocate(count * sizeof(float)),
         output = device.allocate(count * sizeof(float));
    for (uint32_t i = 0; i < count; ++i)
        input.data<float>()[i] = float(i) / 2;
    Graph produce, consume;
    record(produce, producer, kernels.affine, input.view(), shared.view(), {count, 2, 1});
    record(consume, consumer, kernels.affine, shared.view(), output.view(), {count, 3, -2});
    produce.prepare(&producer);
    consume.prepare(&consumer);
    produce.launch(&producer);
    auto first = producer.signal(event);
    consume.launch(&consumer, {first});
    check(producer.statistics().stream_wait_calls == 0 &&
              consumer.statistics().stream_wait_calls == 0,
          "event submission performed a CPU stream wait");
    consumer.synchronize();
    for (uint32_t i = 0; i < count; ++i)
        near(output.data<float>()[i], 6 * input.data<float>()[i] + 1,
             "cross-stream graph dependency result");
    produce.launch(&producer);
    auto second = producer.signal(event);
    consumer.wait(second);
    consume.launch(&consumer);
    consumer.synchronize();
    producer.synchronize();
    check(second.value() == first.value() + 1 && event.completed_value() >= second.value(),
          "timeline is not monotonic/completed");
    check(first.ready() && second.ready(), "completed event points report not ready");
    check(producer.statistics().event_signals == 2 && consumer.statistics().event_waits == 2,
          "GPU event counters are wrong");
    if (policy == DispatchPolicy::Indirect)
        check(producer.statistics().icb_builds == 1 && consumer.statistics().icb_builds == 1,
              "fresh event values rebuilt otherwise unchanged graphs");
}

void event_validation(Device device, const Kernels& kernels) {
    Stream producer{device}, other{device};
    TimelineEvent event{device};
    auto point = producer.signal(event);
    throws([&] { other.signal(event); }, "multiple producer streams were allowed on one timeline");
    throws([&] { other.wait(EventPoint{}); }, "empty event point was accepted");
    auto input = device.allocate(4), output = device.allocate(4);
    Graph graph;
    record(graph, other, kernels.affine, input.view(), output.view(), {1, 1, 0});
    {
        Graph capture;
        auto scope = capture.record_guard(&producer);
        throws([&] { producer.signal(event); }, "event signal was frozen into a graph");
        throws([&] { producer.wait(point); }, "event wait was frozen into a graph");
        throws([&] { graph.launch(&producer, {point}); },
               "launch wait was frozen into an enclosing graph");
        scope.cancel();
    }
    Device alien_device{detail::make_reference_device()};
    Stream alien{alien_device};
    TimelineEvent alien_event{alien_device};
    auto alien_point = alien.signal(alien_event);
    throws([&] { other.wait(alien_point); }, "cross-device event point was accepted");
    throws([&] { producer.signal(alien_event); }, "cross-device event producer was accepted");
    EventPoint retained;
    {
        TimelineEvent temporary{device};
        retained = producer.signal(temporary);
    }
    other.wait(retained).wait();
    other.synchronize();
    producer.synchronize();
    alien.synchronize();
}

// Gated reference callbacks make in-flight behavior deterministic without
// assuming a Metal dispatch lasts a particular number of milliseconds.
struct Gate {
    Stream& producer;
    Stream* consumer;
    std::promise<void> release_promise, started_promise;
    std::shared_future<void> release = release_promise.get_future().share();
    std::future<void> started = started_promise.get_future();
    std::atomic<bool> first{true};
    bool opened = false;
    explicit Gate(Stream& stream, Stream* second = nullptr) : producer(stream), consumer(second) {}
    void enter() {
        if (first.exchange(false))
            started_promise.set_value();
        release.wait();
    }
    void wait_started() {
        check(started.wait_for(std::chrono::seconds(5)) == std::future_status::ready,
              "gated reference dispatch did not start");
    }
    void open() {
        if (!opened) {
            opened = true;
            release_promise.set_value();
        }
    }
    ~Gate() {
        try {
            open();
        } catch (...) {
        }
        try {
            producer.synchronize();
        } catch (...) {
        }
        if (consumer)
            try {
                consumer->synchronize();
            } catch (...) {
            }
    }
};

void in_flight_updates(Device device) {
    Stream stream{device, DispatchPolicy::Indirect};
    Gate gate{stream};
    auto kernel = device.kernel_file(shader, "graph_affine", [&](const ComputeCommand& command) {
        gate.enter();
        affine_reference()(command);
    });
    auto first = device.allocate(4), second = device.allocate(4), out_first = device.allocate(4),
         out_second = device.allocate(4);
    first.data<float>()[0] = 3;
    second.data<float>()[0] = 7;
    std::weak_ptr<detail::BufferAllocation> retained = first.implementation();
    Graph graph;
    record(graph, stream, kernel, first.view(), out_first.view(), {1, 1, 0});
    graph.prepare(&stream);
    auto old = graph.launch(&stream);
    gate.wait_started();
    auto edit = graph.update();
    edit.set_buffer(0, 0, second.view())
        .set_buffer(0, 1, out_second.view())
        .set_scalar(0, 2, Args{1, 2, 1});
    edit.commit();
    auto updated = graph.launch(&stream);
    graph.reset();
    first = {};
    check(stream.statistics().icb_builds == 2,
          "in-flight graph edit did not create a distinct executable version");
    check(!retained.expired(), "in-flight snapshot released its input early");
    gate.open();
    old.wait();
    updated.wait();
    stream.synchronize();
    near(out_first.data<float>()[0], 3, "old in-flight binding/scalar snapshot changed");
    near(out_second.data<float>()[0], 15, "updated snapshot result");
    check(retained.expired(), "completed reference work retained the discarded graph's input");
}

void event_submission_is_nonblocking(Device device, const Kernels& kernels) {
    Stream producer{device}, consumer{device};
    Gate gate{producer, &consumer};
    TimelineEvent event{device};
    auto input = device.allocate(4), shared = device.allocate(4), output = device.allocate(4);
    input.data<float>()[0] = 3;
    shared.data<float>()[0] = -100;
    auto gated = device.kernel_file(shader, "graph_affine", [&](const ComputeCommand& command) {
        gate.enter();
        affine_reference()(command);
    });
    Graph produce, consume;
    record(produce, producer, gated, input.view(), shared.view(), {1, 2, 1});
    record(consume, consumer, kernels.affine, shared.view(), output.view(), {1, 3, -2});
    produce.launch(&producer);
    gate.wait_started();
    auto point = producer.signal(event);
    auto submitting =
        std::async(std::launch::async, [&] { return consume.launch(&consumer, {point}); });
    if (submitting.wait_for(std::chrono::seconds(5)) != std::future_status::ready) {
        gate.open();
        submitting.wait();
        throw std::runtime_error("GPU event dependency blocked the submitting CPU thread");
    }
    auto done = submitting.get();
    check(!done.ready(), "consumer completed before its blocked producer");
    check(event.completed_value() == 0, "timeline advanced before producer completion");
    gate.open();
    done.wait();
    consumer.synchronize();
    producer.synchronize();
    near(output.data<float>()[0], 19, "pending cross-stream dependency did not order work");
}

void producer_error_reaches_consumer(Device device) {
    Stream producer{device}, consumer{device};
    Gate gate{producer, &consumer};
    TimelineEvent event{device};
    std::atomic<bool> consumer_ran{false};
    auto input = device.allocate(4), shared = device.allocate(4), output = device.allocate(4);
    auto fail = device.kernel_file(shader, "graph_affine", [&](const ComputeCommand&) {
        gate.enter();
        throw std::runtime_error("intentional producer-prefix failure");
    });
    auto observe = device.kernel_file(shader, "graph_affine",
                                      [&](const ComputeCommand&) { consumer_ran = true; });
    Graph produce, consume;
    record(produce, producer, fail, input.view(), shared.view(), {1, 1, 0});
    record(consume, consumer, observe, shared.view(), output.view(), {1, 1, 0});
    produce.launch(&producer);
    gate.wait_started();
    auto point = producer.signal(event);
    auto done = consume.launch(&consumer, {point});
    gate.open();
    throws([&] { point.wait(); }, "event point lost an earlier producer failure");
    throws([&] { done.wait(); }, "consumer completion lost its dependency failure");
    check(!consumer_ran, "reference consumer ran after a failed dependency");
    throws([&] { producer.synchronize(); }, "producer synchronization lost its error");
    throws([&] { consumer.synchronize(); }, "consumer synchronization lost its error");
}
void parallel_callbacks_overlap(Device device, const Kernels& kernels, DispatchPolicy policy) {
    // Both independent callbacks must enter before either can finish. A serial
    // implementation fails within the timeout instead of passing a value-only
    // diamond test. This tests the reference scheduler, not GPU occupancy.
    struct Rendezvous {
        std::mutex mutex;
        std::condition_variable changed;
        size_t entered = 0;
        std::atomic<size_t> finished{0};
    };
    auto rendezvous = std::make_shared<Rendezvous>();
    Stream stream{device, policy};
    auto input = device.allocate(4), x = device.allocate(4), y = device.allocate(4),
         z = device.allocate(4), out = device.allocate(4);
    *input.data<float>() = 3;
    auto branch =
        device.kernel_file(shader, "graph_affine", [rendezvous](const ComputeCommand& command) {
            {
                std::unique_lock<std::mutex> lock(rendezvous->mutex);
                ++rendezvous->entered;
                rendezvous->changed.notify_all();
                check(rendezvous->changed.wait_for(lock, std::chrono::seconds(5),
                                                   [&] { return rendezvous->entered == 2; }),
                      "independent callbacks were serialized instead of allowed to overlap");
            }
            affine_reference()(command);
            rendezvous->finished.fetch_add(1, std::memory_order_release);
        });
    auto join =
        device.kernel_file(shader, "graph_sum", [rendezvous](const ComputeCommand& command) {
            check(rendezvous->finished.load(std::memory_order_acquire) == 2,
                  "dependent join ran before both branches finished");
            *command.buffer<float>(2) = *command.buffer<float>(0) + *command.buffer<float>(1);
        });
    Graph graph;
    auto recording = graph.record_guard(&stream);
    affine(stream, kernels.affine, input.view(), x.view(), {1, 2, 1});
    {
        auto group = stream.parallel();
        affine(stream, branch, x.view(), y.view(), {1, 3, 0});
        affine(stream, branch, x.view(), z.view(), {1, -1, 4});
        group.finish();
    }
    stream.dispatch(join, {1, 1, 1}, {1, 1, 1},
                    {Argument::buffer(0, y.view()), Argument::buffer(1, z.view()),
                     Argument::buffer(2, out.view(), Access::Write),
                     Argument::scalar(3, uint32_t{1})});
    recording.finish();
    execute(graph, stream);
    near(*out.data<float>(), 18, "parallel rendezvous/join output");
}
} // namespace

int main(int argc, char** argv) {
    const bool native = argc > 1 && std::string(argv[1]) == "--metal";
    Device device;
    try {
        device = native ? Device::system_default() : Device::reference();
    } catch (const std::exception& error) {
        std::cout << "SKIP: " << error.what() << '\n';
        return 77;
    }
    Device::set_current(device);
    int passed = 0, failed = 0, skipped = 0;
    auto run = [&](const std::string& name, auto action) {
        try {
            action();
            ++passed;
            std::cout << "PASS " << name << '\n';
        } catch (const std::exception& error) {
            ++failed;
            std::cerr << "FAIL " << name << ": " << error.what() << '\n';
        }
    };
    try {
        Kernels kernels{device};
        std::cout << "Backend: " << device.capabilities().name << '\n';
        for (auto policy :
             {DispatchPolicy::PerKernel, DispatchPolicy::Direct, DispatchPolicy::Indirect}) {
            if (policy == DispatchPolicy::Indirect && !device.capabilities().indirect_compute) {
                ++skipped;
                std::cout << "SKIP indirect policies: no compute ICB support\n";
                continue;
            }
            const std::string prefix = policy_name(policy);
            run(prefix + " record/prepare without execution",
                [&] { recording_and_preparation(device, kernels, policy); });
            run(prefix + " scalar/binding/grid updates",
                [&] { update_bindings_scalars_grid(device, kernels, policy); });
            run(prefix + " independent graph clones",
                [&] { independent_clones(device, kernels, policy); });
            run(prefix + " parallel diamond and transactional hazards",
                [&] { diamond_parallel(device, kernels, policy); });
            run(prefix + " fresh cross-stream event dependencies",
                [&] { cross_stream_events(device, kernels, policy); });
        }
        run("transaction validation and conflicts",
            [&] { transactional_validation(device, kernels); });
        run("parallel alias/scope validation", [&] { parallel_validation(device, kernels); });
        run("timeline producer/capture/device validation",
            [&] { event_validation(device, kernels); });
        if (device.capabilities().backend == BackendKind::Reference) {
            run("deterministic in-flight update snapshots", [&] { in_flight_updates(device); });
            run("pending GPU dependency does not block CPU submit",
                [&] { event_submission_is_nonblocking(device, kernels); });
            run("producer-prefix errors reach dependent consumers",
                [&] { producer_error_reaches_consumer(device); });
            run("Direct independent callbacks overlap and join",
                [&] { parallel_callbacks_overlap(device, kernels, DispatchPolicy::Direct); });
            run("Indirect independent callbacks overlap and join",
                [&] { parallel_callbacks_overlap(device, kernels, DispatchPolicy::Indirect); });
        } else {
            skipped += 5;
            std::cout << "SKIP 5 CPU-only scheduling/fault-injection cases; native diamond outputs "
                         "are tested above\n";
        }
    } catch (const std::exception& error) {
        ++failed;
        std::cerr << "FAIL setup: " << error.what() << '\n';
    }
    std::cout << passed << " passed, " << skipped << " skipped, " << failed << " failed\n";
    return failed ? 1 : 0;
}
