#include <metalrt/control.h>
#include <algorithm>
#include <iostream>
#include <string>

using namespace metalrt;
namespace {
struct Counts {
    uint32_t x, y, z;
};
struct WriteConfig {
    uint32_t capacity, width, height, value;
};
} // namespace
int main(int argc, char** argv) {
    try {
        bool reference = false;
        DispatchPolicy policy = DispatchPolicy::Indirect;
        for (int i = 1; i < argc; ++i) {
            const std::string argument = argv[i];
            if (argument == "--reference")
                reference = true;
            else if (argument == "--direct")
                policy = DispatchPolicy::Direct;
            else if (argument == "--per-kernel")
                policy = DispatchPolicy::PerKernel;
            else
                throw std::invalid_argument(
                    "usage: gpu_control [--reference] [--direct|--per-kernel]");
        }
        auto device = reference ? Device::reference() : Device::system_default();
        Device::set_current(device);
        Stream stream(device, policy);
        Graph graph;
        const std::string file = std::string(METALRT_EXAMPLE_SHADER_DIR) + "/gpu_control.metal";
        auto counts_kernel =
            device.kernel_file(file, "control_counts", [](const ComputeCommand& c) {
                const auto value = c.scalar<Counts>(1);
                std::memcpy(c.buffer<uint32_t>(0), &value, sizeof(value));
            });
        auto store_kernel = device.kernel_file(file, "control_store", [](const ComputeCommand& c) {
            *c.buffer<uint32_t>(0) = c.scalar<uint32_t>(1);
        });
        auto decrement = device.kernel_file(file, "control_decrement", [](const ComputeCommand& c) {
            auto& value = *c.buffer<uint32_t>(0);
            if (value)
                --value;
        });
        auto write = device.kernel_file(file, "control_write", [](const ComputeCommand& c) {
            const auto config = c.scalar<WriteConfig>(1);
            auto* out = c.buffer<uint32_t>(0);
            for (size_t i = 0; i < std::min(c.grid.x, size_t(config.capacity)); ++i)
                out[i] = config.value + uint32_t(i);
        });
        auto gpu_counts = device.allocate(12, StorageMode::Private);
        auto condition = device.allocate(4), remaining = device.allocate(4),
             branch = device.allocate(4);
        auto output = device.allocate(32 * 4), status = device.allocate(4);
        auto store = [&](BufferView destination, uint32_t value) {
            stream.dispatch(
                store_kernel, {1, 1, 1}, {1, 1, 1},
                {Argument::buffer(0, destination, Access::Write), Argument::scalar(1, value)});
        };

        auto recording = graph.record_guard(&stream); // No execution while building the plan.
        stream.dispatch(counts_kernel, {1, 1, 1}, {1, 1, 1},
                        {Argument::buffer(0, gpu_counts.view(), Access::Write),
                         Argument::scalar(1, Counts{3, 1, 1})});
        store(condition.view(), 1);
        store(remaining.view(), 3);
        if_else(
            stream, {condition.view()}, [&] { store(branch.view(), 11); },
            [&] { store(branch.view(), 22); });
        auto loop = repeat_while(stream, {remaining.view()}, 8, [&](uint32_t) {
            stream.dispatch(decrement, {1, 1, 1}, {1, 1, 1},
                            {Argument::buffer(0, remaining.view(), Access::ReadWrite)});
        });
        // Counts are GPU-produced THREADGROUP counts. The payload has 8 threads
        // per group and at most 4 groups, so the stable output needs 32 elements.
        stream.dispatch_indirect(write, gpu_counts.view(), {4, 1, 1}, {8, 1, 1},
                                 {Argument::buffer(0, output.view(), Access::Write),
                                  Argument::scalar(1, WriteConfig{32, 32, 1, 100})},
                                 status.view());
        recording.finish();
        graph.prepare(&stream);

        for (size_t iteration = 0; iteration < 3; ++iteration) {
            std::fill_n(output.data<uint32_t>(), 32, 0u); // Prior replay has completed below.
            auto ticket = graph.launch(&stream);          // No CPU reads between GPU stages.
            const auto loop_status = loop.read(ticket);   // One explicit final readback boundary.
            if (*branch.data<uint32_t>() != 11 || loop_status.iterations != 3 ||
                loop_status.limit_reached || *status.data<uint32_t>() != 0)
                throw std::runtime_error("GPU branch/loop/status verification failed");
            for (uint32_t i = 0; i < 32; ++i)
                if (output.data<uint32_t>()[i] != (i < 24 ? 100 + i : 0))
                    throw std::runtime_error("GPU-sized dispatch verification failed");
            std::cout << "Replay " << iteration
                      << ": branch 11, loop iterations 3, indirect payload threads 24\n";
        }
        const auto statistics = stream.statistics();
        std::cout << "Backend: " << device.capabilities().name
                  << "\nICB builds/reuses/updates: " << statistics.icb_builds << '/'
                  << statistics.icb_reuses << '/' << statistics.icb_updates
                  << "\nRecorded payload dispatches: " << statistics.compute_dispatches
                  << "\nRuntime control dispatches: " << statistics.control_dispatches << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
