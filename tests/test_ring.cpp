#include <metalrt/parameters.h>
#include <metalrt/detail/backend.h>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <future>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <vector>

using namespace metalrt;
namespace {
void check(bool value, const char* message) {
    if (!value)
        throw std::runtime_error(message);
}
void near(float value, float expected) {
    if (!std::isfinite(value) || std::abs(value - expected) > 1e-5f)
        throw std::runtime_error("ring numerical mismatch");
}
template <class F> void throws(F&& function, const char* message) {
    bool failed = false;
    try {
        function();
    } catch (const std::exception&) {
        failed = true;
    }
    check(failed, message);
}
struct RingArgs {
    uint32_t count, stride_floats;
    float scale, bias;
};
struct Parameters {
    uint32_t count;
    float scale, bias;
    uint32_t pad = 0;
};
std::string shader(const char* name) {
    return std::string(METALRT_TEST_SHADER_DIR) + "/" + name;
}
Kernel per_slot_kernel(Device device) {
    return device.kernel_file(shader("ring_pipeline.metal"), "ring_per_slot",
                              [](const ComputeCommand& c) {
                                  auto args = c.scalar<RingArgs>(1);
                                  auto* values = c.buffer<float>(0);
                                  for (uint32_t i = 0; i < args.count; ++i)
                                      values[args.count + i] = values[i] * args.scale + args.bias;
                              });
}
Kernel indexed_kernel(Device device, std::shared_ptr<std::atomic<bool>> fail_once = {}) {
    return device.kernel_file(
        shader("ring_pipeline.metal"), "ring_indexed", [fail_once](const ComputeCommand& c) {
            if (fail_once && fail_once->exchange(false))
                throw std::runtime_error("intentional reference ring failure");
            auto args = c.scalar<RingArgs>(2);
            auto* values = c.buffer<float>(0);
            auto slot = *c.buffer<uint32_t>(1);
            const auto base = size_t(slot) * args.stride_floats;
            for (uint32_t i = 0; i < args.count; ++i)
                values[base + args.count + i] = values[base + i] * args.scale + args.bias;
        });
}
RingPipeline::Builder builder(Kernel per_slot, Kernel indexed, uint32_t count) {
    return [=](Stream& stream, const RingBindings& b) {
        RingArgs args{count, static_cast<uint32_t>(b.stride_bytes / sizeof(float)), 2, 0.25f};
        if (b.indexed())
            launch_linear(&stream, indexed, count,
                          {Argument::buffer(0, b.storage, Access::ReadWrite),
                           Argument::buffer(1, b.slot_index), Argument::scalar(2, args)});
        else
            launch_linear(
                &stream, per_slot, count,
                {Argument::buffer(0, b.storage, Access::ReadWrite), Argument::scalar(1, args)});
    };
}
void pipeline_reuse(Device device, DispatchPolicy policy, RingBindingMode mode, size_t slots) {
    constexpr uint32_t count = 65;
    constexpr size_t iterations = 19;
    RingPipeline pipeline{device, policy,
                          slots,  2 * count * sizeof(float),
                          mode,   builder(per_slot_kernel(device), indexed_kernel(device), count)};
    const auto prepared = pipeline.stream_statistics();
    const auto prepared_allocations = device.statistics().buffer_allocations;
    check(pipeline.graph_count() == (mode == RingBindingMode::Indexed ? 1 : slots),
          "incorrect prepared ring variant count");
    check(prepared.compute_dispatches == 0,
          "preparing a ring must not execute an uninitialized slot");
    if (policy == DispatchPolicy::Indirect)
        check(prepared.icb_builds == pipeline.graph_count(),
              "expected one ICB per prepared variant");
    // Abandoning a CPU preparation does not skip an indexed GPU schedule entry.
    {
        auto unused = pipeline.acquire();
        check(unused.slot() == 0, "first slot");
    }
    std::vector<int> previous(slots, -1);
    for (size_t iteration = 0; iteration < iterations; ++iteration) {
        auto lease = pipeline.acquire();
        const size_t slot = lease.slot();
        check(slot == iteration % slots, "ring order drifted");
        auto* data = lease.data_as<float>();
        if (previous[slot] >= 0)
            for (uint32_t i = 0; i < count; ++i)
                near(data[count + i], 2 * (previous[slot] + float(i) / 16) + 0.25f);
        for (uint32_t i = 0; i < count; ++i)
            data[i] = float(iteration) + float(i) / 16;
        previous[slot] = static_cast<int>(iteration);
        lease.submit();
        throws([&] { lease.data(); }, "submitted lease must not allow CPU access");
    }
    pipeline.drain();
    for (size_t slot = 0; slot < slots; ++slot) {
        const auto bytes = pipeline.readback(slot);
        std::vector<float> values(2 * count);
        std::memcpy(values.data(), bytes.data(), bytes.size());
        for (uint32_t i = 0; i < count; ++i)
            near(values[count + i], 2 * (previous[slot] + float(i) / 16) + 0.25f);
    }
    auto stats = pipeline.stream_statistics();
    check(device.statistics().buffer_allocations == prepared_allocations,
          "fixed ring replay allocated another GPU buffer");
    check(stats.control_dispatches == (mode == RingBindingMode::Indexed ? iterations : 0),
          "indexed advance must be accounted as control work");
    if (policy == DispatchPolicy::Indirect) {
        check(stats.icb_builds == prepared.icb_builds, "ring replay rebuilt an ICB");
        check(stats.icb_updates == 0, "ring replay rewrote an ICB");
        check(stats.icb_reuses == iterations, "every ring submission should reuse a prepared ICB");
    }
    check(pipeline.ring_statistics().submissions == iterations, "ring submission count");
    pipeline.recover();
    auto restarted = pipeline.acquire();
    check(restarted.slot() == 0, "recovery must reset CPU cursor");
    auto* data = restarted.data_as<float>();
    for (uint32_t i = 0; i < count; ++i)
        data[i] = 3;
    restarted.submit().wait();
    auto bytes = pipeline.readback(0);
    std::vector<float> values(2 * count);
    std::memcpy(values.data(), bytes.data(), bytes.size());
    near(values[count], 6.25f);
}
void stable_parameters(Device device, DispatchPolicy policy) {
    Stream stream{device, policy};
    constexpr uint32_t count = 67;
    auto input = device.allocate(count * sizeof(float)),
         output = device.allocate(count * sizeof(float));
    for (uint32_t i = 0; i < count; ++i)
        input.data<float>()[i] = float(i) / 8;
    ParameterBuffer<Parameters> parameters{device, Parameters{count, 1, 0}};
    const auto original = parameters.view();
    auto kernel = device.kernel_file(shader("stable_parameters.metal"), "stable_affine",
                                     [](const ComputeCommand& c) {
                                         const auto p = *c.buffer<const Parameters>(2);
                                         const auto* x = c.buffer<float>(0);
                                         auto* y = c.buffer<float>(1);
                                         for (uint32_t i = 0; i < p.count; ++i)
                                             y[i] = p.scale * x[i] + p.bias;
                                     });
    Graph graph;
    {
        auto record = graph.record_guard(&stream);
        launch_linear(&stream, kernel, count,
                      {Argument::buffer(0, input.view()),
                       Argument::buffer(1, output.view(), Access::Write), parameters.bind(2)});
        record.finish();
    }
    graph.prepare(&stream);
    for (uint32_t run = 0; run < 8; ++run) {
        const Parameters p{count, 0.5f + run, float(run) / 8};
        parameters.submit(stream, graph, p).wait();
        for (uint32_t i = 0; i < count; ++i)
            near(output.data<float>()[i], p.scale * input.data<float>()[i] + p.bias);
        check(parameters.view() == original, "parameter contents update changed buffer identity");
    }
    const auto p = parameters.get();
    near(p.scale, 7.5f);
    {
        auto edit = parameters.edit();
        edit.value().scale = 2;
        edit.value().bias = -1;
        edit.launch(stream, graph).wait();
    }
    near(output.data<float>()[1], -0.75f);
    const auto stats = stream.statistics();
    if (policy == DispatchPolicy::Indirect)
        check(stats.icb_builds == 1 && stats.icb_updates == 0 && stats.icb_reuses == 9,
              "parameter-buffer updates must reuse one ICB");
}
void lease_validation(Device device) {
    RingBuffer ring{device, 2, 64};
    Stream stream{device};
    throws([&] { RingBuffer invalid{device, 0, 64}; }, "zero slot count must fail");
    throws([&] { RingBuffer invalid{device, 2, SIZE_MAX}; }, "ring size overflow must fail");
    throws([&] { RingBuffer invalid{device, 2, 64, 512}; }, "unsupported slot alignment must fail");
    throws([&] { ring.binding(2); }, "invalid ring binding index must fail");
    Graph graph;
    {
        auto record = graph.record_guard(&stream);
        stream.fill(ring.binding(0), 7);
        record.finish();
    }
    auto lease = ring.acquire();
    check(!ring.try_acquire(), "only one CPU producer lease is permitted");
    throws([&] { ring.acquire(); }, "same-thread second acquire must not deadlock");
    throws([&] { ring.drain(); }, "drain while CPU owns a slot must fail");
    throws([&] { ring.readback(0); }, "readback while CPU owns a slot must fail");
    throws([&] { ring.recover(); }, "recover while CPU owns a slot must fail");
    auto moved = std::move(lease);
    throws([&] { lease.data(); }, "moved-from lease must be invalid");
    bool wrong_thread_rejected = false;
    std::thread wrong_thread([&] {
        try {
            moved.data();
        } catch (const std::logic_error&) {
            wrong_thread_rejected = true;
        }
    });
    wrong_thread.join();
    check(wrong_thread_rejected, "CPU lease must enforce its acquiring thread");
    auto foreign = Device{detail::make_reference_device()};
    Stream foreign_stream{foreign};
    throws([&] { moved.launch(foreign_stream, graph); }, "cross-device ring submission must fail");
    check(static_cast<bool>(moved), "validation error must preserve CPU lease");
    {
        auto record = stream.batch();
        throws([&] { moved.launch(stream, graph); }, "submitting into an outer capture must fail");
        record.cancel();
    }
    moved.launch(stream, graph).wait();
    check(!moved, "launch must consume lease");
    auto bytes = ring.readback(0);
    for (auto value : bytes)
        check(value == 7, "submitted slot contents");
}
struct Gate {
    std::promise<void> entered_promise, release_promise;
    std::future<void> entered = entered_promise.get_future();
    std::shared_future<void> release = release_promise.get_future().share();
    std::atomic<bool> released{false};
    void open() {
        if (!released.exchange(true))
            release_promise.set_value();
    }
    ~Gate() { open(); }
};
void blocked_slot_waits() {
    auto device = Device::reference();
    Stream stream{device};
    auto gate = std::make_shared<Gate>();
    // Always open the gate before destroying the stream, including assertion failures.
    struct Opener {
        std::shared_ptr<Gate> gate;
        ~Opener() { gate->open(); }
    } opener{gate};
    auto blocker = device.kernel("reference-only", "ring_blocker", [gate](const ComputeCommand& c) {
        gate->entered_promise.set_value();
        gate->release.wait();
        *c.buffer<uint32_t>(0) = 42;
    });
    RingBuffer ring{device, 2, 16};
    {
        auto lease = ring.acquire();
        lease.submit(stream, [&](Stream& s, BufferView slot) {
            launch_linear(&s, blocker, 1, {Argument::buffer(0, slot, Access::Write)});
        });
    }
    check(gate->entered.wait_for(std::chrono::seconds(5)) == std::future_status::ready,
          "reference worker never entered blocking kernel");
    // Slot 1 is available while slot 0 is blocked; no device-wide wait is allowed.
    {
        auto lease = ring.try_acquire();
        check(static_cast<bool>(lease) && lease->slot() == 1,
              "next free slot should not wait for unrelated GPU use");
        lease->submit(stream, [](Stream& s, BufferView slot) { s.fill(slot, 3); });
    }
    check(!ring.try_acquire(), "wraparound must not expose a busy slot");
    gate->open();
    ring.drain();
    auto lease = ring.acquire();
    check(lease.slot() == 0 && *lease.data_as<uint32_t>() == 42,
          "wrapped slot is synchronized before CPU access");
    check(ring.statistics().try_misses > 0, "busy try-acquire should be observable");
}
void blocked_parameters() {
    const auto device = Device::reference();
    Stream stream{device};
    auto gate = std::make_shared<Gate>();
    struct Opener {
        std::shared_ptr<Gate> gate;
        ~Opener() { gate->open(); }
    } opener{gate};
    ParameterBuffer<uint32_t> parameters{device, 13};
    auto result = device.allocate(sizeof(uint32_t));
    auto kernel =
        device.kernel("reference-only", "parameters_blocker", [gate](const ComputeCommand& c) {
            gate->entered_promise.set_value();
            gate->release.wait();
            *c.buffer<uint32_t>(1) = *c.buffer<const uint32_t>(0);
        });
    Graph graph;
    {
        auto record = graph.record_guard(&stream);
        launch_linear(&stream, kernel, 1,
                      {parameters.bind(0), Argument::buffer(1, result.view(), Access::Write)});
        record.finish();
    }
    const auto ticket = parameters.launch(stream, graph);
    check(gate->entered.wait_for(std::chrono::seconds(5)) == std::future_status::ready,
          "parameter worker never entered blocking kernel");
    check(!parameters.try_edit(), "parameter edit must not expose storage still read by GPU");
    gate->open();
    ticket.wait();
    check(*result.data<uint32_t>() == 13, "parameter contents changed during GPU read");
    {
        auto edit = parameters.try_edit();
        check(static_cast<bool>(edit), "completed parameter storage should be editable");
        edit->value() = 99;
    }
    check(parameters.get() == 99, "typed parameter edit was not preserved");
}
void failure_and_recovery() {
    const auto device = Device::reference();
    auto fail = std::make_shared<std::atomic<bool>>(true);
    RingPipeline pipeline{device,
                          DispatchPolicy::Indirect,
                          2,
                          16,
                          RingBindingMode::Indexed,
                          builder(per_slot_kernel(device), indexed_kernel(device, fail), 2)};
    {
        auto lease = pipeline.acquire();
        auto* x = lease.data_as<float>();
        x[0] = 1;
        x[1] = 2;
        lease.submit();
    }
    throws([&] { pipeline.drain(); }, "GPU failure must surface on drain");
    check(pipeline.poisoned(), "failed ring must be poisoned");
    throws([&] { pipeline.acquire(); }, "poisoned ring must not expose slots");
    pipeline.recover();
    check(!pipeline.poisoned(), "recovery should clear observed failure after draining");
    {
        auto lease = pipeline.acquire();
        check(lease.slot() == 0, "recovered schedule must start at zero");
        auto* x = lease.data_as<float>();
        x[0] = 3;
        x[1] = 4;
        lease.submit().wait();
    }
    const auto bytes = pipeline.readback(0);
    float values[4];
    std::memcpy(values, bytes.data(), sizeof(values));
    near(values[2], 6.25f);
    near(values[3], 8.25f);
    // A CPU exception after a deliberately flushed prefix cannot silently recycle a slot.
    RingBuffer ring{device, 2, 16};
    Stream stream{device};
    auto lease = ring.acquire();
    throws(
        [&] {
            lease.submit(stream, [](Stream& s, BufferView slot) {
                s.fill(slot, 9);
                s.host([] {});
                throw std::runtime_error("after prefix");
            });
        },
        "encoding callback error must surface");
    check(ring.poisoned() && !lease, "failed handoff must consume and poison the lease");
    ring.recover();
    stream.synchronize();
    auto after = ring.acquire();
    check(after.slot() == 0, "failed recording must not advance scheduling");
}
void failed_callback_retains_busy_slots() {
    const auto device = Device::reference();
    Stream stream{device};
    auto gate = std::make_shared<Gate>();
    struct Opener {
        std::shared_ptr<Gate> gate;
        ~Opener() { gate->open(); }
    } opener{gate};
    RingBuffer ring{device, 2, 16};
    auto blocker =
        device.kernel("reference-only", "ring_failure_prefix", [gate](const ComputeCommand& c) {
            gate->entered_promise.set_value();
            gate->release.wait();
            *c.buffer<uint32_t>(0) = 42;
        });
    {
        auto lease = ring.acquire();
        lease.submit(stream, [&](Stream& s, BufferView slot) {
            launch_linear(&s, blocker, 1, {Argument::buffer(0, slot, Access::Write)});
        });
    }
    check(gate->entered.wait_for(std::chrono::seconds(5)) == std::future_status::ready,
          "failed-callback worker did not start");
    {
        auto lease = ring.acquire();
        throws(
            [&] {
                lease.submit(stream, [](Stream&, BufferView) {
                    throw std::runtime_error("cancel second slot");
                });
            },
            "callback failure should propagate");
    }
    check(ring.poisoned(), "callback failure must poison all scheduling");
    auto recovery = std::async(std::launch::async, [&] { ring.recover(); });
    // This guard opens before the async future joins if an assertion throws.
    Opener unblock_recovery{gate};
    check(recovery.wait_for(std::chrono::milliseconds(20)) == std::future_status::timeout,
          "recovery released an allocation while GPU still used it");
    gate->open();
    recovery.get();
    stream.synchronize();
    auto lease = ring.acquire();
    check(lease.slot() == 0 && *lease.data_as<uint32_t>() == 42,
          "recovery must wait and reset the complete ring");
}
void lifetime_and_moves() {
    const auto device = Device::reference();
    Stream stream{device};
    auto gate = std::make_shared<Gate>();
    struct Opener {
        std::shared_ptr<Gate> gate;
        ~Opener() { gate->open(); }
    } opener{gate};
    auto blocker =
        device.kernel("reference-only", "ring_lifetime", [gate](const ComputeCommand& c) {
            gate->entered_promise.set_value();
            gate->release.wait();
            *c.buffer<uint32_t>(0) = 77;
        });
    std::weak_ptr<detail::BufferAllocation> allocation;
    {
        auto ring = std::make_unique<RingBuffer>(device, 2, 16);
        allocation = ring->binding(0).buffer.implementation();
        auto lease = ring->acquire();
        ring.reset(); // The lease retains its state and allocation.
        *lease.data_as<uint32_t>() = 0;
        lease.submit(stream, [&](Stream& s, BufferView b) {
            launch_linear(&s, blocker, 1, {Argument::buffer(0, b, Access::Write)});
        });
    }
    check(gate->entered.wait_for(std::chrono::seconds(5)) == std::future_status::ready,
          "lifetime worker did not start");
    check(!allocation.expired(), "submitted commands must retain destroyed ring storage");
    gate->open();
    stream.synchronize();
    check(allocation.expired(), "finished one-shot ring work should release unused storage");
    RingBuffer first{device, 2, 16};
    auto lease = first.acquire();
    RingBuffer moved{std::move(first)};
    *lease.data_as<uint32_t>() = 5;
    lease.submit(stream, [](Stream& s, BufferView b) { s.fill(b, 0); }).wait();
    auto next = moved.acquire();
    check(next.slot() == 1, "moving owner must preserve slot schedule");
}
} // namespace
int main(int argc, char** argv) {
    try {
        const bool native = argc > 1 && std::string(argv[1]) == "--metal";
        Device device;
        if (native) {
            try {
                device = Device::system_default();
            } catch (const std::exception& e) {
                std::cout << "SKIP: " << e.what() << '\n';
                return 77;
            }
        } else
            device = Device::reference();
        Device::set_current(device);
        size_t cases = 0, skipped = 0;
        for (auto policy :
             {DispatchPolicy::PerKernel, DispatchPolicy::Direct, DispatchPolicy::Indirect}) {
            if (policy == DispatchPolicy::Indirect && !device.capabilities().indirect_compute) {
                ++skipped;
                std::cout << "SKIP unsupported indirect ring policy\n";
                continue;
            }
            for (auto mode : {RingBindingMode::PerSlot, RingBindingMode::Indexed}) {
                pipeline_reuse(device, policy, mode, 3);
                ++cases;
            }
            pipeline_reuse(device, policy, RingBindingMode::Indexed, 2);
            ++cases;
            stable_parameters(device, policy);
            ++cases;
        }
        lease_validation(device);
        ++cases;
        if (!native) {
            blocked_slot_waits();
            blocked_parameters();
            failure_and_recovery();
            failed_callback_retains_busy_slots();
            lifetime_and_moves();
            cases += 5;
        } else {
            skipped += 5;
            std::cout << "SKIP reference-only deterministic worker/failure injections\n";
        }
        std::cout << "PASS ring/parameters: " << cases << " cases, " << skipped << " skipped on "
                  << device.capabilities().name << '\n';
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "RING TEST FAILURE: " << e.what() << '\n';
        return 1;
    }
}
