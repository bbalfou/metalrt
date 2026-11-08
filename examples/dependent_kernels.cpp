// Standalone runtime example: no tiny-metal-nn dependency and no generated source.
#include <metalrt/runtime.h>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>

using namespace metalrt;
namespace {
constexpr float pi = 3.14159265358979323846f;
// Handwritten kernels use a bounds check because dispatch rounds up threadgroups.
#ifndef METALRT_EXAMPLE_SHADER_DIR
#define METALRT_EXAMPLE_SHADER_DIR "kernels"
#endif
const std::string shader_file =
    std::string{METALRT_EXAMPLE_SHADER_DIR} + "/dependent_kernels.metal";
struct Shape {
    uint32_t count;
};
DispatchPolicy policy_from(const std::string& name) {
    if (name == "per-kernel")
        return DispatchPolicy::PerKernel;
    if (name == "direct")
        return DispatchPolicy::Direct;
    if (name == "indirect")
        return DispatchPolicy::Indirect;
    throw std::invalid_argument("policy must be per-kernel, direct, or indirect");
}
} // namespace
int main(int argc, char** argv) {
    try {
        bool reference = false;
        std::string name = "direct", shader_path = shader_file;
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg == "--reference")
                reference = true;
            else if (arg == "--policy" && i + 1 < argc)
                name = argv[++i];
            else if (arg == "--shader" && i + 1 < argc)
                shader_path = argv[++i];
            else
                throw std::invalid_argument("usage: dependent_kernels [--reference] [--policy "
                                            "per-kernel|direct|indirect] [--shader file.metal]");
        }
        Device device = reference ? Device::reference() : Device::system_default();
        Device::set_current(device);
        Stream stream{device, policy_from(name)};
        constexpr uint32_t n = 257; // Deliberately not a multiple of the threadgroup size.
        auto input = device.allocate(n * 2 * sizeof(float));
        auto encoded = device.allocate(n * 4 * sizeof(float));
        auto weights = device.allocate(3 * 4 * sizeof(float));
        auto hidden = device.allocate(n * 3 * sizeof(float));
        auto output = device.allocate(n * sizeof(float));
        for (uint32_t i = 0; i < n; ++i) {
            input.data<float>()[2 * i] = float(i) / n;
            input.data<float>()[2 * i + 1] = float(n - i) / n;
        }
        for (uint32_t i = 0; i < 12; ++i)
            weights.data<float>()[i] = (float(i) - 5.0f) / 10;
        auto encode =
            device.kernel_file(shader_path, "encode_features", [](const ComputeCommand& c) {
                auto a = c.scalar<Shape>(2);
                auto* x = c.buffer<const float>(0);
                auto* y = c.buffer<float>(1);
                for (uint32_t i = 0; i < a.count; ++i) {
                    y[4 * i] = x[2 * i];
                    y[4 * i + 1] = x[2 * i + 1];
                    y[4 * i + 2] = std::sin(pi * x[2 * i]);
                    y[4 * i + 3] = std::cos(pi * x[2 * i + 1]);
                }
            });
        auto dense = device.kernel_file(shader_path, "dense_layer", [](const ComputeCommand& c) {
            auto a = c.scalar<Shape>(3);
            auto* x = c.buffer<const float>(0);
            auto* w = c.buffer<const float>(1);
            auto* y = c.buffer<float>(2);
            for (uint32_t i = 0; i < a.count; ++i)
                for (uint32_t k = 0; k < 3; ++k) {
                    float sum = 0;
                    for (uint32_t j = 0; j < 4; ++j)
                        sum += w[k * 4 + j] * x[i * 4 + j];
                    y[i * 3 + k] = sum;
                }
        });
        auto reduce = device.kernel_file(shader_path, "relu_reduce", [](const ComputeCommand& c) {
            auto a = c.scalar<Shape>(2);
            auto* x = c.buffer<const float>(0);
            auto* y = c.buffer<float>(1);
            for (uint32_t i = 0; i < a.count; ++i)
                y[i] = std::max(0.0f, x[3 * i]) + std::max(0.0f, x[3 * i + 1]) +
                       std::max(0.0f, x[3 * i + 2]);
        });
        Graph graph;
        {
            auto capture = graph.capture_guard(&stream);
            launch_linear(&stream, encode, n,
                          {Argument::buffer(0, input.view()),
                           Argument::buffer(1, encoded.view(), Access::Write),
                           Argument::scalar(2, Shape{n})});
            // The MLP reads the SAME encoded allocation. No staging or copy is involved.
            launch_linear(&stream, dense, n * 3,
                          {Argument::buffer(0, encoded.view()), Argument::buffer(1, weights.view()),
                           Argument::buffer(2, hidden.view(), Access::Write),
                           Argument::scalar(3, Shape{n})});
            launch_linear(&stream, reduce, n,
                          {Argument::buffer(0, hidden.view()),
                           Argument::buffer(1, output.view(), Access::Write),
                           Argument::scalar(2, Shape{n})});
            capture.finish(); // Destructor also submits; finish() reports synchronous errors here.
        }
        stream.synchronize();
        auto commands = graph.commands();
        if (!(commands[0].compute.argument(1).resource == commands[1].compute.argument(0).resource))
            throw std::runtime_error("encoding/MLP buffer alias was lost");
        // CPU logic runs once when recording. Explicit graph replay reruns GPU work only.
        graph.launch(&stream).wait();
        float maximum_error = 0;
        for (uint32_t i = 0; i < n; ++i) {
            const float x = input.data<float>()[2 * i], y = input.data<float>()[2 * i + 1];
            const float features[4] = {x, y, std::sin(pi * x), std::cos(pi * y)};
            float expected = 0;
            for (uint32_t k = 0; k < 3; ++k) {
                float sum = 0;
                for (uint32_t j = 0; j < 4; ++j)
                    sum += weights.data<float>()[k * 4 + j] * features[j];
                expected += std::max(0.0f, sum);
            }
            maximum_error = std::max(maximum_error, std::abs(expected - output.data<float>()[i]));
        }
        if (maximum_error > 2e-5f)
            throw std::runtime_error("GPU/reference mismatch");
        const auto stats = stream.statistics();
        std::cout << "Backend: " << device.capabilities().name << "\nPolicy: " << name
                  << "\nVerified " << n << " samples; max absolute error " << maximum_error
                  << "\nEncoding and MLP share one buffer: yes"
                  << "\nTwo executions: " << stats.compute_dispatches << " kernels, "
                  << stats.command_buffers << " command buffers"
                  << "\nICB builds/reuses/updates: " << stats.icb_builds << '/' << stats.icb_reuses
                  << '/' << stats.icb_updates << '\n';
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
