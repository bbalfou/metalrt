#include <metalrt/runtime.h>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

#ifndef METALRT_TEST_SHADER_DIR
#define METALRT_TEST_SHADER_DIR "kernels"
#endif
using namespace metalrt;
namespace {
struct Args {
    uint32_t count;
    float scale, bias;
};
static_assert(sizeof(Args) == 12, "MSL constant ABI");
void check(bool condition, const char* message) {
    if (!condition)
        throw std::runtime_error(message);
}
template <class F> void throws(F&& function, const char* message) {
    try {
        function();
    } catch (const std::exception&) {
        return;
    }
    throw std::runtime_error(message);
}
CpuKernel reference() {
    return [](const ComputeCommand& command) {
        auto args = command.scalar<Args>(2);
        auto* input = command.buffer<const float>(0);
        auto* output = command.buffer<float>(1);
        for (uint32_t i = 0; i < args.count; ++i)
            output[i] = input[i] * args.scale + args.bias;
    };
}
struct TemporaryDirectory {
    std::filesystem::path path;
    TemporaryDirectory() {
        auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
        // CTest runs in a writable build directory, including restricted
        // environments with no global /tmp directory.
        path = std::filesystem::current_path() / ("metalrt-shader-files-" + std::to_string(stamp));
        std::filesystem::create_directory(path);
    }
    ~TemporaryDirectory() {
        std::error_code error;
        std::filesystem::remove_all(path, error);
    }
};
void exercise(Device device, const std::filesystem::path& path, DispatchPolicy policy,
              bool remove_source = false) {
    constexpr uint32_t count = 37;
    auto kernel = device.kernel_file(path.string(), "file_affine", reference());
    if (remove_source)
        check(std::filesystem::remove(path), "temporary shader source was not removed");
    auto input = device.allocate(count * sizeof(float)),
         output = device.allocate(count * sizeof(float));
    for (uint32_t i = 0; i < count; ++i)
        input.data<float>()[i] = float(i) - 10;
    Stream stream{device, policy};
    Graph graph;
    {
        auto capture = graph.capture_guard(&stream);
        launch_linear(&stream, kernel, count,
                      {Argument::buffer(0, input.view()),
                       Argument::buffer(1, output.view(), Access::Write),
                       Argument::scalar(2, Args{count, 1.5f, -2.0f})});
        capture.finish();
    }
    graph.launch(&stream).wait();
    stream.synchronize();
    for (uint32_t i = 0; i < count; ++i)
        check(output.data<float>()[i] == input.data<float>()[i] * 1.5f - 2.0f,
              "file-loaded kernel output mismatch");
    check(stream.statistics().compute_dispatches == 2, "file-loaded graph should execute twice");
    if (policy == DispatchPolicy::Indirect)
        check(stream.statistics().icb_builds == 1 && stream.statistics().icb_reuses == 1,
              "file-loaded ICB must be reusable");
}
} // namespace
int main(int argc, char** argv) {
    try {
        Device device;
        if (argc > 1 && std::string(argv[1]) == "--metal") {
            try {
                device = Device::system_default();
            } catch (const std::exception& error) {
                std::cout << "SKIP: " << error.what() << '\n';
                return 77;
            }
        } else
            device = Device::reference();
        Device::set_current(device);
        const auto source = std::filesystem::path{METALRT_TEST_SHADER_DIR} / "file_dispatch.metal";
        check(std::filesystem::is_regular_file(source),
              "CMake did not copy the handwritten .metal asset");
        size_t passed = 0, skipped = 0;
        for (auto policy :
             {DispatchPolicy::PerKernel, DispatchPolicy::Direct, DispatchPolicy::Indirect}) {
            if (policy == DispatchPolicy::Indirect && !device.capabilities().indirect_compute) {
                ++skipped;
                continue;
            }
            exercise(device, source, policy);
            ++passed;
        }
        TemporaryDirectory temporary;
        auto spaced = temporary.path / "source with spaces.metal";
        std::filesystem::copy_file(source, spaced);
        // The compiled kernel must own what it needs: dispatch and replay stay
        // valid after the caller's original source asset has been removed.
        exercise(device, spaced, DispatchPolicy::Direct, true);
        ++passed;
        throws(
            [&] {
                device.kernel_file((temporary.path / "missing.metal").string(), "file_affine",
                                   reference());
            },
            "missing shader file must fail");
        ++passed;
        auto empty = temporary.path / "empty.metal";
        std::ofstream{empty}.close();
        throws([&] { device.kernel_file(empty.string(), "file_affine", reference()); },
               "empty shader file must fail");
        ++passed;
        throws([&] { device.kernel_file(temporary.path.string(), "file_affine", reference()); },
               "directory must not be accepted as shader source");
        ++passed;
        std::cout << "PASS shader files: " << passed << " cases, " << skipped << " skipped on "
                  << device.capabilities().name << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "SHADER FILE TEST FAILURE: " << error.what() << '\n';
        return 1;
    }
}
