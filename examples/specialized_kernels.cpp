#include "example_support.h"

// The same source and entry produce two native function-constant specializations.
// Pass --metallib PATH to avoid runtime source compilation entirely for this shader.
int main(int argc, char** argv) {
    using namespace metalrt_examples;
    try {
        bool reference = false;
        std::string library;
        for (int i = 1; i < argc; ++i) {
            if (std::string(argv[i]) == "--reference")
                reference = true;
            else if (std::string(argv[i]) == "--metallib" && i + 1 < argc)
                library = argv[++i];
            else
                throw std::invalid_argument(
                    "usage: specialized_kernels [--reference] [--metallib PATH]");
        }
        auto device = reference ? Device::reference() : Device::system_default();
        Stream stream{device};
        auto output = device.allocate(256 * sizeof(float));
        for (float scale : {2.0f, 5.0f}) {
            KernelCompileOptions options;
            options.constants = {
                FunctionConstant::scalar(0, scale), FunctionConstant::scalar(1, uint32_t{3}),
                FunctionConstant::scalar(2, true), FunctionConstant::scalar(3, int32_t{-1})};
            CpuKernel cpu = [scale](const ComputeCommand& c) {
                for (size_t i = 0; i < c.grid.x; ++i)
                    c.buffer<float>(0)[i] = float(i) * scale + 2;
            };
            auto kernel = library.empty()
                              ? device.kernel_file(std::string(METALRT_EXAMPLE_SHADER_DIR) +
                                                       "/function_constants.metal",
                                                   "specialized_value", options, cpu)
                              : device.kernel_metallib(library, "specialized_value", options, cpu);
            GpuTimingOptions timing;
            timing.sample_kernels = device.capabilities().kernel_timestamps;
            if (timing.sample_kernels && device.capabilities().statistic_counters)
                timing.counters = GpuCounterSet::Statistic;
            stream.begin_gpu_timing(timing);
            stream.dispatch(kernel, {256, 1, 1}, {64, 1, 1},
                            {Argument::buffer(0, output.view(), Access::Write)});
            const auto measured = stream.end_gpu_timing();
            near(output.data<float>()[255], 255 * scale + 2);
            std::cout << "scale=" << scale << ", output[255]=" << output.data<float>()[255] << '\n';
            for (const auto& sample : measured.kernels) {
                if (sample.valid)
                    std::cout << sample.kernel << ": " << sample.seconds * 1e6 << " us\n";
                for (const auto& counter : sample.counters)
                    if (counter.valid())
                        std::cout << "  " << counter.name << ": " << counter.begin << " -> "
                                  << counter.end << '\n';
            }
            if (measured.valid())
                std::cout << "Occupied GPU interval: " << measured.occupied_seconds * 1e6
                          << " us\n";
        }
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
