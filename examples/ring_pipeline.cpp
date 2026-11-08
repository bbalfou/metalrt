#include <metalrt/ring.h>
#include <cmath>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

using namespace metalrt;
namespace {
struct RingArgs {
    uint32_t count, stride_floats;
    float scale, bias;
};
static_assert(sizeof(RingArgs) == 16, "MSL argument ABI");
void near(float value, float expected) {
    if (std::abs(value - expected) > 1e-5f)
        throw std::runtime_error("ring output mismatch");
}
} // namespace
int main(int argc, char** argv) {
    try {
        bool reference = false;
        RingBindingMode mode = RingBindingMode::Indexed;
        size_t slots = 3;
        DispatchPolicy policy = DispatchPolicy::Indirect;
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--reference")
                reference = true;
            else if (arg == "--per-slot")
                mode = RingBindingMode::PerSlot;
            else if (arg == "--double")
                slots = 2;
            else if (arg == "--direct")
                policy = DispatchPolicy::Direct;
            else if (arg == "--per-kernel")
                policy = DispatchPolicy::PerKernel;
            else
                throw std::invalid_argument("usage: ring_pipeline [--reference] [--per-slot] "
                                            "[--double] [--direct|--per-kernel]");
        }
        auto device = reference ? Device::reference() : Device::system_default();
        Device::set_current(device);
        constexpr uint32_t count = 65;
        const std::string path = std::string(METALRT_EXAMPLE_SHADER_DIR) + "/ring_pipeline.metal";
        auto per_slot = device.kernel_file(path, "ring_per_slot", [](const ComputeCommand& c) {
            const auto args = c.scalar<RingArgs>(1);
            auto* data = c.buffer<float>(0);
            for (uint32_t i = 0; i < args.count; ++i)
                data[args.count + i] = data[i] * args.scale + args.bias;
        });
        auto indexed = device.kernel_file(path, "ring_indexed", [](const ComputeCommand& c) {
            const auto args = c.scalar<RingArgs>(2);
            auto* data = c.buffer<float>(0);
            const auto slot = *c.buffer<uint32_t>(1);
            const auto base = size_t(slot) * args.stride_floats;
            for (uint32_t i = 0; i < args.count; ++i)
                data[base + args.count + i] = data[base + i] * args.scale + args.bias;
        });
        RingPipeline pipeline{
            device,
            policy,
            slots,
            2 * count * sizeof(float),
            mode,
            [&](Stream& stream, const RingBindings& binding) {
                const RingArgs args{count,
                                    static_cast<uint32_t>(binding.stride_bytes / sizeof(float)),
                                    2.0f, 0.25f};
                if (binding.indexed())
                    launch_linear(&stream, indexed, count,
                                  {Argument::buffer(0, binding.storage, Access::ReadWrite),
                                   Argument::buffer(1, binding.slot_index),
                                   Argument::scalar(2, args)});
                else
                    launch_linear(&stream, per_slot, count,
                                  {Argument::buffer(0, binding.storage, Access::ReadWrite),
                                   Argument::scalar(1, args)});
            }};
        std::vector<int> last(slots, -1);
        for (int iteration = 0; iteration < 24; ++iteration) {
            auto lease = pipeline.acquire(); // Only this slot's preceding ticket is waited.
            const size_t slot = lease.slot();
            auto* mapped = lease.data_as<float>();
            if (last[slot] >= 0)
                for (uint32_t i = 0; i < count; ++i)
                    near(mapped[count + i], 2.0f * (last[slot] + float(i) / 100) + 0.25f);
            for (uint32_t i = 0; i < count; ++i)
                mapped[i] = iteration + float(i) / 100;
            last[slot] = iteration;
            lease.submit(); // CPU can now fill the next slot while this GPU work runs.
        }
        pipeline.drain();
        for (size_t slot = 0; slot < slots; ++slot) {
            const auto bytes = pipeline.readback(slot);
            std::vector<float> values(2 * count);
            std::memcpy(values.data(), bytes.data(), bytes.size());
            for (uint32_t i = 0; i < count; ++i)
                near(values[count + i], 2.0f * (last[slot] + float(i) / 100) + 0.25f);
        }
        const auto stats = pipeline.stream_statistics();
        const char* policy_name = policy == DispatchPolicy::Indirect ? "Indirect"
                                  : policy == DispatchPolicy::Direct ? "Direct"
                                                                     : "PerKernel";
        std::cout << "Backend: " << device.capabilities().name << "\nPolicy: " << policy_name
                  << "\nSlots: " << slots << "\nPrepared graphs: " << pipeline.graph_count()
                  << "\nVerified 24 iterations without recapture"
                  << "\nICB builds/reuses/updates: " << stats.icb_builds << '/' << stats.icb_reuses
                  << '/' << stats.icb_updates
                  << "\nSlot waits: " << pipeline.ring_statistics().slot_waits << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
