#include <metalrt/parameters.h>
#include <cmath>
#include <iostream>
#include <string>

using namespace metalrt;
struct Parameters {
    uint32_t count;
    float scale, bias;
    uint32_t pad = 0;
};
static_assert(sizeof(Parameters) == 16, "MSL constant-buffer ABI");
int main(int argc, char** argv) {
    try {
        bool reference = false;
        DispatchPolicy policy = DispatchPolicy::Indirect;
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--reference")
                reference = true;
            else if (arg == "--direct")
                policy = DispatchPolicy::Direct;
            else if (arg == "--per-kernel")
                policy = DispatchPolicy::PerKernel;
            else
                throw std::invalid_argument(
                    "usage: stable_parameters [--reference] [--direct|--per-kernel]");
        }
        auto device = reference ? Device::reference() : Device::system_default();
        Device::set_current(device);
        Stream stream{device, policy};
        constexpr uint32_t count = 257;
        auto input = device.allocate(count * sizeof(float)),
             output = device.allocate(count * sizeof(float));
        for (uint32_t i = 0; i < count; ++i)
            input.data<float>()[i] = float(i) / 32;
        ParameterBuffer<Parameters> parameters{device, Parameters{count, 1, 0}};
        auto kernel =
            device.kernel_file(std::string(METALRT_EXAMPLE_SHADER_DIR) + "/stable_parameters.metal",
                               "stable_affine", [](const ComputeCommand& command) {
                                   const auto args = *command.buffer<const Parameters>(2);
                                   const auto* x = command.buffer<const float>(0);
                                   auto* y = command.buffer<float>(1);
                                   for (uint32_t i = 0; i < args.count; ++i)
                                       y[i] = x[i] * args.scale + args.bias;
                               });
        Graph graph;
        {
            auto recording = graph.record_guard(&stream);
            launch_linear(&stream, kernel, count,
                          {Argument::buffer(0, input.view()),
                           Argument::buffer(1, output.view(), Access::Write), parameters.bind(2)});
            recording.finish();
        }
        graph.prepare(&stream);
        for (uint32_t iteration = 0; iteration < 12; ++iteration) {
            const Parameters value{count, 0.5f + iteration, float(iteration) / 10};
            parameters.submit(stream, graph, value).wait();
            for (uint32_t i = 0; i < count; ++i)
                if (std::abs(output.data<float>()[i] -
                             (input.data<float>()[i] * value.scale + value.bias)) > 1e-5f)
                    throw std::runtime_error("parameter-buffer replay mismatch");
        }
        const auto stats = stream.statistics();
        const char* policy_name = policy == DispatchPolicy::Indirect ? "Indirect"
                                  : policy == DispatchPolicy::Direct ? "Direct"
                                                                     : "PerKernel";
        std::cout << "Backend: " << device.capabilities().name << "\nPolicy: " << policy_name
                  << "\n12 parameter changes with one stable binding"
                  << "\nICB builds/reuses/updates: " << stats.icb_builds << '/' << stats.icb_reuses
                  << '/' << stats.icb_updates << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
