#pragma once
#include <metalrt/runtime.h>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>

#ifndef METALRT_EXAMPLE_SHADER_DIR
#define METALRT_EXAMPLE_SHADER_DIR "kernels"
#endif

namespace metalrt_examples {
using namespace metalrt;
struct AffineArgs {
    uint32_t count;
    float scale, bias;
};
static_assert(sizeof(AffineArgs) == 12, "graph_features.metal constant ABI");
struct Settings {
    bool reference = false;
    DispatchPolicy policy = DispatchPolicy::Indirect;
};
inline Settings settings(int argc, char** argv, const char* program) {
    Settings result;
    for (int i = 1; i < argc; ++i) {
        const std::string argument = argv[i];
        if (argument == "--reference")
            result.reference = true;
        else if (argument == "--policy" && i + 1 < argc) {
            const std::string policy = argv[++i];
            if (policy == "direct")
                result.policy = DispatchPolicy::Direct;
            else if (policy == "per-kernel")
                result.policy = DispatchPolicy::PerKernel;
            else if (policy == "indirect")
                result.policy = DispatchPolicy::Indirect;
            else
                throw std::invalid_argument("policy must be per-kernel, direct, or indirect");
        } else
            throw std::invalid_argument(std::string("usage: ") + program +
                                        " [--reference] [--policy per-kernel|direct|indirect]");
    }
    return result;
}
inline Device select_device(const Settings& settings) {
    auto device = settings.reference ? Device::reference() : Device::system_default();
    Device::set_current(device);
    std::cout << "Backend: " << device.capabilities().name << '\n';
    return device;
}
inline Kernel affine_kernel(Device device) {
    return device.kernel_file(std::string{METALRT_EXAMPLE_SHADER_DIR} + "/graph_features.metal",
                              "graph_affine", [](const ComputeCommand& command) {
                                  const auto args = command.scalar<AffineArgs>(2);
                                  const auto* input = command.buffer<const float>(0);
                                  auto* output = command.buffer<float>(1);
                                  for (size_t i = 0;
                                       i < std::min<size_t>(args.count, command.grid.x); ++i)
                                      output[i] = input[i] * args.scale + args.bias;
                              });
}
inline Kernel sum_kernel(Device device) {
    return device.kernel_file(std::string{METALRT_EXAMPLE_SHADER_DIR} + "/graph_features.metal",
                              "graph_sum", [](const ComputeCommand& command) {
                                  const auto count = command.scalar<uint32_t>(3);
                                  const auto* a = command.buffer<const float>(0);
                                  const auto* b = command.buffer<const float>(1);
                                  auto* output = command.buffer<float>(2);
                                  for (size_t i = 0; i < std::min<size_t>(count, command.grid.x);
                                       ++i)
                                      output[i] = a[i] + b[i];
                              });
}
inline void affine(Stream& stream, const Kernel& kernel, BufferView input, BufferView output,
                   uint32_t count, float scale, float bias) {
    launch_linear(&stream, kernel, count,
                  {Argument::buffer(0, std::move(input)),
                   Argument::buffer(1, std::move(output), Access::Write),
                   Argument::scalar(2, AffineArgs{count, scale, bias})});
}
inline void near(float actual, float expected) {
    if (!std::isfinite(actual) ||
        std::abs(actual - expected) > 1e-5f * std::max(1.0f, std::abs(expected)))
        throw std::runtime_error("result mismatch: " + std::to_string(actual) + " versus " +
                                 std::to_string(expected));
}
} // namespace metalrt_examples
