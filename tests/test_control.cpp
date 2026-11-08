#include <metalrt/control.h>
#include <metalrt/detail/backend.h>
#include <algorithm>
#include <iostream>
#include <mutex>
#include <string>

using namespace metalrt;
namespace {
void check(bool okay, const std::string& message) {
    if (!okay)
        throw std::runtime_error(message);
}
template <class F> void throws(F&& f, const char* message) {
    bool caught = false;
    try {
        f();
    } catch (const std::exception&) {
        caught = true;
    }
    check(caught, message);
}
struct WriteConfig {
    uint32_t capacity, width, height, value;
};
struct Counts {
    uint32_t x, y, z;
};
static_assert(sizeof(WriteConfig) == 16 && sizeof(Counts) == 12, "GPU control example ABI");
#ifndef METALRT_TEST_SHADER_DIR
#define METALRT_TEST_SHADER_DIR "kernels"
#endif
struct Kernels {
    Kernel write, store, increment, decrement, counts, atomic_add;
    explicit Kernels(Device device) {
        const std::string file = std::string(METALRT_TEST_SHADER_DIR) + "/gpu_control.metal";
        write = device.kernel_file(file, "control_write", [](const ComputeCommand& c) {
            const auto config = c.scalar<WriteConfig>(1);
            auto* output = c.buffer<uint32_t>(0);
            for (size_t z = 0; z < c.grid.z; ++z)
                for (size_t y = 0; y < c.grid.y; ++y)
                    for (size_t x = 0; x < c.grid.x; ++x) {
                        const size_t linear = x + config.width * (y + config.height * z);
                        if (linear < config.capacity)
                            output[linear] = config.value + static_cast<uint32_t>(linear);
                    }
        });
        store = device.kernel_file(file, "control_store", [](const ComputeCommand& c) {
            *c.buffer<uint32_t>(0) = c.scalar<uint32_t>(1);
        });
        increment = device.kernel_file(file, "control_increment", [](const ComputeCommand& c) {
            *c.buffer<uint32_t>(0) += c.scalar<uint32_t>(1);
        });
        decrement = device.kernel_file(file, "control_decrement", [](const ComputeCommand& c) {
            auto& value = *c.buffer<uint32_t>(0);
            if (value)
                --value;
        });
        counts = device.kernel_file(file, "control_counts", [](const ComputeCommand& c) {
            auto value = c.scalar<Counts>(1);
            std::memcpy(c.buffer<uint32_t>(0), &value, sizeof(value));
        });
        auto lock = std::make_shared<std::mutex>();
        atomic_add =
            device.kernel_file(file, "control_atomic_add", [lock](const ComputeCommand& c) {
                std::lock_guard<std::mutex> guard(*lock);
                *c.buffer<uint32_t>(0) += static_cast<uint32_t>(c.grid.x * c.grid.y * c.grid.z);
            });
    }
};
void store(Stream& stream, const Kernels& kernels, BufferView output, uint32_t value) {
    stream.dispatch(kernels.store, {1, 1, 1}, {1, 1, 1},
                    {Argument::buffer(0, output, Access::Write), Argument::scalar(1, value)});
}
void increment(Stream& stream, const Kernels& kernels, BufferView output, uint32_t value = 1) {
    stream.dispatch(kernels.increment, {1, 1, 1}, {1, 1, 1},
                    {Argument::buffer(0, output, Access::ReadWrite), Argument::scalar(1, value)});
}
void decrement(Stream& stream, const Kernels& kernels, BufferView value) {
    stream.dispatch(kernels.decrement, {1, 1, 1}, {1, 1, 1},
                    {Argument::buffer(0, value, Access::ReadWrite)});
}
void write_counts(Stream& stream, const Kernels& kernels, BufferView output, Counts values) {
    stream.dispatch(kernels.counts, {1, 1, 1}, {1, 1, 1},
                    {Argument::buffer(0, output, Access::Write), Argument::scalar(1, values)});
}
std::vector<Argument> write_args(Buffer output, WriteConfig config) {
    return {Argument::buffer(0, output.view(), Access::Write), Argument::scalar(1, config)};
}

void indirect_bounds(Device device, const Kernels& kernels, DispatchPolicy policy) {
    Stream stream(device, policy);
    Graph graph;
    auto counts = device.allocate(12, StorageMode::Private), output = device.allocate(32 * 4),
         status = device.allocate(4);
    auto record = graph.record_guard(&stream);
    write_counts(stream, kernels, counts.view(), {2, 1, 1}); // GPU producer, no CPU readback.
    stream.dispatch_indirect(kernels.write, counts.view(), {4, 1, 1}, {8, 1, 1},
                             write_args(output, {32, 32, 1, 100}), status.view());
    record.finish();
    graph.prepare(&stream);
    check(stream.statistics().command_buffers == 0, "prepare must not execute control work");
    check(!stream.last_ticket(), "prepare must not manufacture a completion ticket");
    for (auto requested : {0u, 1u, 4u, 5u, UINT32_MAX}) {
        std::fill_n(output.data<uint32_t>(), 32, 0u);
        auto edit = graph.update();
        edit.set_scalar(0, 1, Counts{requested, 1, 1});
        edit.commit();
        graph.launch(&stream).wait();
        const auto written = std::min(requested, 4u) * 8;
        for (uint32_t i = 0; i < 32; ++i)
            check(output.data<uint32_t>()[i] == (i < written ? 100 + i : 0),
                  "indirect count clamp or zero-dispatch mismatch");
        check(*status.data<uint32_t>() == (requested > 4 ? 1u : 0u), "indirect clamp status");
    }
    check(stream.statistics().control_dispatches >= 5,
          "indirect preparation is counted separately");
}
void indirect_3d(Device device, const Kernels& kernels, DispatchPolicy policy) {
    Stream stream(device, policy);
    auto counts = device.allocate(12), output = device.allocate(96 * 4),
         status = device.allocate(4);
    const Counts request{2, 1, 2};
    std::memcpy(counts.contents(), &request, 12);
    std::fill_n(output.data<uint32_t>(), 96, 0u);
    stream.dispatch_indirect(kernels.write, counts.view(), {3, 2, 2}, {4, 2, 1},
                             write_args(output, {96, 12, 4, 7}), status.view());
    stream.synchronize();
    for (uint32_t z = 0; z < 2; ++z)
        for (uint32_t y = 0; y < 4; ++y)
            for (uint32_t x = 0; x < 12; ++x) {
                const auto index = x + 12 * (y + 4 * z);
                check(output.data<uint32_t>()[index] == (x < 8 && y < 2 ? 7 + index : 0),
                      "three-dimensional indirect grid");
            }
    check(*status.data<uint32_t>() == 0, "3D dispatch status");
}
void indirect_predicate(Device device, const Kernels& kernels, DispatchPolicy policy) {
    Stream stream(device, policy);
    Graph graph;
    auto counts = device.allocate(12), predicate = device.allocate(4),
         output = device.allocate(32 * 4), status = device.allocate(4);
    Counts request{99, 1, 1};
    std::memcpy(counts.contents(), &request, 12);
    *predicate.data<uint32_t>() = 0;
    std::fill_n(output.data<uint32_t>(), 32, 0u);
    auto recording = graph.record_guard(&stream);
    {
        auto scope = stream.predicate_scope(predicate.view());
        stream.dispatch_indirect(kernels.write, counts.view(), {4, 1, 1}, {8, 1, 1},
                                 write_args(output, {32, 32, 1, 3}), status.view());
        scope.finish();
    }
    recording.finish();
    graph.launch(&stream).wait();
    check(*status.data<uint32_t>() == 3, "inactive and clamped status bits combine");
    for (uint32_t i = 0; i < 32; ++i)
        check(output.data<uint32_t>()[i] == 0, "inactive indirect payload executed");
    *predicate.data<uint32_t>() = 1;
    graph.launch(&stream).wait();
    check(output.data<uint32_t>()[31] == 34 && *status.data<uint32_t>() == 1,
          "active indirect replay");
}
void predicate_update(Device device, const Kernels& kernels) {
    Stream stream(device, DispatchPolicy::Indirect);
    Graph graph;
    auto counts = device.allocate(12), off = device.allocate(4), on = device.allocate(4),
         output = device.allocate(32 * 4), status = device.allocate(4);
    const Counts request{1, 1, 1};
    std::memcpy(counts.contents(), &request, 12);
    *off.data<uint32_t>() = 0;
    *on.data<uint32_t>() = 1;
    std::fill_n(output.data<uint32_t>(), 32, 0u);
    auto recording = graph.record_guard(&stream);
    {
        auto scope = stream.predicate_scope(off.view());
        stream.dispatch_indirect(kernels.write, counts.view(), {4, 1, 1}, {8, 1, 1},
                                 write_args(output, {32, 32, 1, 9}), status.view());
        scope.finish();
    }
    recording.finish();
    graph.launch(&stream).wait();
    check(*status.data<uint32_t>() == 2, "initial predicate should suppress work");
    graph.update().set_predicate(0, on.view()).commit();
    graph.launch(&stream).wait();
    check(output.data<uint32_t>()[7] == 16 && *status.data<uint32_t>() == 0,
          "transactional predicate rebind");
    graph.update().set_predicate(0, {}).commit();
    *on.data<uint32_t>() = 0;
    output.data<uint32_t>()[7] = 0;
    graph.launch(&stream).wait();
    check(output.data<uint32_t>()[7] == 16,
          "removing a user predicate should preserve indirect dispatch");
    throws([&] { graph.update().set_indirect_groups(0, {}, {4, 1, 1}).commit(); },
           "empty indirect-group edit must not silently become static");
}
void branch_snapshot(Device device, const Kernels& kernels, DispatchPolicy policy) {
    Stream stream(device, policy);
    Graph graph;
    auto flag = device.allocate(4), output = device.allocate(4);
    *flag.data<uint32_t>() = 1;
    *output.data<uint32_t>() = 0;
    auto recording = graph.record_guard(&stream);
    if_else(
        stream, {flag.view()},
        [&] {
            store(stream, kernels, flag.view(), 0);
            store(stream, kernels, output.view(), 11);
        },
        [&] { store(stream, kernels, output.view(), 22); });
    recording.finish();
    check(*output.data<uint32_t>() == 0, "recording a conditional must not execute it");
    graph.launch(&stream).wait();
    check(*flag.data<uint32_t>() == 0 && *output.data<uint32_t>() == 11,
          "branch must snapshot condition before true branch mutates it");
    graph.launch(&stream).wait();
    check(*output.data<uint32_t>() == 22, "branch chooses else on next replay");
}
void comparisons(Device device, const Kernels& kernels, DispatchPolicy policy) {
    Stream stream(device, policy);
    auto flag = device.allocate(4), output = device.allocate(4);
    *flag.data<uint32_t>() = 2;
    const bool expected[] = {true, false, false, true, true, true, false, false};
    for (uint32_t operation = 0; operation < 8; ++operation) {
        Graph graph;
        auto recording = graph.record_guard(&stream);
        if_else(
            stream, {flag.view(), static_cast<Compare>(operation), 3},
            [&] { store(stream, kernels, output.view(), 1); },
            [&] { store(stream, kernels, output.view(), 0); });
        recording.finish();
        graph.launch(&stream).wait();
        check(*output.data<uint32_t>() == uint32_t(expected[operation]),
              "unsigned GPU predicate comparison");
    }
}
void bounded_loop(Device device, const Kernels& kernels, DispatchPolicy policy) {
    Stream stream(device, policy);
    Graph graph;
    auto remaining = device.allocate(4), output = device.allocate(4);
    auto gpu_result_copy = device.allocate(8);
    auto recording = graph.record_guard(&stream);
    auto result = repeat_while(stream, {remaining.view()}, 5, [&](uint32_t) {
        decrement(stream, kernels, remaining.view());
        increment(stream, kernels, output.view());
    });
    stream.copy(result.iterations(), gpu_result_copy.view(0, 4));
    stream.copy(result.limit_reached(), gpu_result_copy.view(4, 4));
    recording.finish();
    graph.prepare(&stream);
    for (uint32_t initial : {0u, 3u, 5u, 7u}) {
        *remaining.data<uint32_t>() = initial;
        *output.data<uint32_t>() = 0;
        const auto status = result.read(graph.launch(&stream));
        check(status.iterations == std::min(initial, 5u), "loop iteration count");
        check(status.limit_reached == (initial > 5),
              "loop limit reports remaining work, not mere equality to bound");
        check(gpu_result_copy.data<uint32_t>()[0] == status.iterations &&
                  gpu_result_copy.data<uint32_t>()[1] == uint32_t(status.limit_reached),
              "downstream GPU commands must consume loop results without a CPU boundary");
        check(*output.data<uint32_t>() == status.iterations &&
                  *remaining.data<uint32_t>() == initial - status.iterations,
              "inactive loop body must not execute");
    }
}
void nested_control(Device device, const Kernels& kernels, DispatchPolicy policy) {
    Stream stream(device, policy);
    Graph graph;
    auto parent = device.allocate(4), child = device.allocate(4), remaining = device.allocate(4),
         output = device.allocate(4);
    LoopResult nested;
    auto recording = graph.record_guard(&stream);
    if_else(
        stream, {parent.view()},
        [&] {
            if_else(
                stream, {child.view()}, [&] { increment(stream, kernels, output.view(), 10); },
                [&] { increment(stream, kernels, output.view(), 20); });
            nested = repeat_while(stream, {remaining.view()}, 4, [&](uint32_t) {
                decrement(stream, kernels, remaining.view());
                increment(stream, kernels, output.view());
            });
        },
        [&] { increment(stream, kernels, output.view(), 1); });
    recording.finish();
    for (uint32_t outer : {0u, 1u})
        for (uint32_t inner : {0u, 1u}) {
            *parent.data<uint32_t>() = outer;
            *child.data<uint32_t>() = inner;
            *remaining.data<uint32_t>() = 2;
            *output.data<uint32_t>() = 0;
            const auto status = nested.read(graph.launch(&stream));
            check(*output.data<uint32_t>() == (outer ? (inner ? 12u : 22u) : 1u),
                  "nested branch must include parent gate");
            check(status.iterations == (outer ? 2u : 0u) && !status.limit_reached,
                  "inactive nested loop must report zero work");
        }
}
void transactional_recording(Device device, const Kernels& kernels) {
    Stream stream(device);
    Graph graph;
    auto flag = device.allocate(4), output = device.allocate(4);
    *flag.data<uint32_t>() = 1;
    *output.data<uint32_t>() = 0;
    auto recording = graph.record_guard(&stream);
    store(stream, kernels, output.view(), 1);
    throws(
        [&] {
            if_else(
                stream, {flag.view()}, [&] { store(stream, kernels, output.view(), 99); },
                [] { throw std::runtime_error("intentional callback failure"); });
        },
        "recording callback should throw");
    throws(
        [&] {
            repeat_while(stream, {flag.view()}, 3, [&](uint32_t i) {
                increment(stream, kernels, output.view());
                if (i == 1)
                    throw std::runtime_error("loop failure");
            });
        },
        "loop callback should throw");
    increment(stream, kernels, output.view());
    recording.finish();
    check(graph.size() == 2, "failed helpers must roll back their complete recorded suffix");
    graph.launch(&stream).wait();
    check(*output.data<uint32_t>() == 2, "failed helper payload leaked into capture");
}
void invalid_controls(Device device, const Kernels& kernels) {
    Stream stream(device);
    auto buffer = device.allocate(128);
    auto counts = buffer.view(0, 12), status = buffer.view(64, 4);
    throws([&] { if_else(stream, {buffer.view(2, 4)}, [] {}); }, "misaligned condition");
    throws([&] { repeat_while(stream, {buffer.view(0, 4)}, 0, [](uint32_t) {}); },
           "zero loop bound");
    throws([&] { stream.dispatch_indirect(kernels.write, counts, {0, 1, 1}, {1, 1, 1}, {}); },
           "missing indirect bound");
    throws(
        [&] {
            stream.dispatch_indirect(kernels.write, buffer.view(0, 8), {1, 1, 1}, {1, 1, 1}, {});
        },
        "short count buffer");
    throws(
        [&] { stream.dispatch_indirect(kernels.write, counts, {UINT32_MAX, 1, 1}, {2, 1, 1}, {}); },
        "overflow indirect grid");
    throws(
        [&] {
            stream.dispatch_indirect(kernels.write, counts, {1, 1, 1}, {1, 1, 1}, {},
                                     counts.slice(0, 4));
        },
        "count/status alias");
    throws(
        [&] {
            stream.dispatch_indirect(kernels.write, counts, {1, 1, 1}, {1, 1, 1},
                                     {Argument::buffer(0, status, Access::Write)}, status);
        },
        "payload/status alias");
    throws(
        [&] { if_else(stream, {buffer.view(0, 4)}, [&] { stream.fill(buffer.view(16, 4), 1); }); },
        "conditional blit must reject, not execute unconditionally");
    throws(
        [&] {
            auto group = stream.parallel();
            stream.dispatch_indirect(kernels.write, counts, {1, 1, 1}, {1, 1, 1}, {});
        },
        "dynamic parallel group unsupported");
    Device other{detail::make_reference_device()};
    auto alien = other.allocate(4);
    throws([&] { if_else(stream, {alien.view()}, [] {}); }, "cross-device condition");
    check(!stream.is_recording() && stream.statistics().command_buffers == 0,
          "invalid helper must restore recording state");
}
void dynamic_update_reuses_static(Device device, const Kernels& kernels) {
    Stream stream(device, DispatchPolicy::Indirect);
    Graph graph;
    auto first = device.allocate(4), last = device.allocate(4), output = device.allocate(32 * 4),
         counts = device.allocate(12), replacement = device.allocate(12);
    const Counts a{1, 1, 1}, b{2, 1, 1};
    std::memcpy(counts.contents(), &a, 12);
    std::memcpy(replacement.contents(), &b, 12);
    auto recording = graph.record_guard(&stream);
    store(stream, kernels, first.view(), 7);
    stream.dispatch_indirect(kernels.write, counts.view(), {4, 1, 1}, {8, 1, 1},
                             write_args(output, {32, 32, 1, 8}));
    store(stream, kernels, last.view(), 9);
    recording.finish();
    graph.prepare(&stream);
    graph.launch(&stream).wait();
    stream.reset_statistics();
    auto edit = graph.update();
    edit.set_indirect_groups(1, replacement.view(), {3, 1, 1});
    edit.commit();
    graph.launch(&stream).wait();
    const auto stats = stream.statistics();
    check(stats.icb_builds == 0 && stats.icb_updates == 0 && stats.icb_reuses == 2,
          "editing only a dynamic node must retain adjacent static ICB segments");
    check(*first.data<uint32_t>() == 7 && *last.data<uint32_t>() == 9 &&
              output.data<uint32_t>()[15] == 23,
          "updated dynamic binding output");
}
void concurrent_dynamic_plan(Device device, const Kernels& kernels) {
    Stream a(device, DispatchPolicy::Indirect), b(device, DispatchPolicy::Indirect);
    Graph graph;
    auto counts = device.allocate(12), counter = device.allocate(4);
    const Counts request{3, 1, 1};
    std::memcpy(counts.contents(), &request, 12);
    *counter.data<uint32_t>() = 0;
    auto recording = graph.record_guard(&a);
    stream_or_default(&a).dispatch_indirect(
        kernels.atomic_add, counts.view(), {4, 1, 1}, {8, 1, 1},
        {Argument::buffer(0, counter.view(), Access::ReadWrite)});
    recording.finish();
    graph.prepare(&a);
    for (size_t i = 0; i < 32; ++i) {
        graph.launch(&a);
        graph.launch(&b);
    }
    a.synchronize();
    b.synchronize();
    check(*counter.data<uint32_t>() == 64u * 24u,
          "concurrent replays must own independent indirect-argument scratch; payload uses "
          "explicit atomics");
}
} // namespace

int main(int argc, char** argv) {
    bool native = false;
    for (int i = 1; i < argc; ++i)
        if (std::string(argv[i]) == "--metal")
            native = true;
        else
            return 2;
    Device device;
    try {
        device = native ? Device::system_default() : Device::reference();
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return native ? 77 : 1;
    }
    if (!device.capabilities().indirect_compute)
        return 77;
    Device::set_current(device);
    size_t passed = 0, failed = 0;
    try {
        Kernels kernels(device);
        auto run = [&](const std::string& name, const std::function<void()>& test) {
            try {
                test();
                ++passed;
                std::cout << "PASS " << name << '\n';
            } catch (const std::exception& error) {
                ++failed;
                std::cerr << "FAIL " << name << ": " << error.what() << '\n';
            }
        };
        for (auto policy :
             {DispatchPolicy::PerKernel, DispatchPolicy::Direct, DispatchPolicy::Indirect}) {
            const auto suffix = policy == DispatchPolicy::PerKernel ? "/PerKernel"
                                : policy == DispatchPolicy::Direct  ? "/Direct"
                                                                    : "/Indirect";
            run(std::string("GPU count clamp and zero work") + suffix,
                [&] { indirect_bounds(device, kernels, policy); });
            run(std::string("3D indirect groups") + suffix,
                [&] { indirect_3d(device, kernels, policy); });
            run(std::string("predicate plus clamp status") + suffix,
                [&] { indirect_predicate(device, kernels, policy); });
            run(std::string("branch snapshots") + suffix,
                [&] { branch_snapshot(device, kernels, policy); });
            run(std::string("unsigned comparisons") + suffix,
                [&] { comparisons(device, kernels, policy); });
            run(std::string("bounded loop and exhaustion") + suffix,
                [&] { bounded_loop(device, kernels, policy); });
            run(std::string("nested control flow") + suffix,
                [&] { nested_control(device, kernels, policy); });
        }
        run("transactional helper failure", [&] { transactional_recording(device, kernels); });
        run("validation and rollback", [&] { invalid_controls(device, kernels); });
        run("dynamic edits reuse static ICBs",
            [&] { dynamic_update_reuses_static(device, kernels); });
        run("transactional predicate edits", [&] { predicate_update(device, kernels); });
        run("concurrent dynamic plan scratch", [&] { concurrent_dynamic_plan(device, kernels); });
    } catch (const std::exception& error) {
        ++failed;
        std::cerr << error.what() << '\n';
    }
    std::cout << "control tests: " << passed << " passed, " << failed << " failed\n";
    return failed ? 1 : 0;
}
