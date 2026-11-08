#include "image_io.h"
#include <metalrt/runtime.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>

using namespace metalrt;
using Clock = std::chrono::steady_clock;
namespace {
struct UV {
    float u, v;
};
struct Dimensions {
    uint32_t width, height, count;
};
double seconds(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}
} // namespace
int main(int argc, char** argv) {
    try {
        std::string path = "data/images/Albert_Einstein_Head.jpg";
        uint32_t count = 1u << 20, iterations = 50;
        DispatchPolicy policy = DispatchPolicy::Direct;
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg == "--image" && i + 1 < argc)
                path = argv[++i];
            else if (arg == "--samples" && i + 1 < argc)
                count = metalrt_examples::positive_count(argv[++i]);
            else if (arg == "--iterations" && i + 1 < argc)
                iterations = metalrt_examples::positive_count(argv[++i]);
            else if (arg == "--indirect")
                policy = DispatchPolicy::Indirect;
            else
                throw std::invalid_argument("usage: image_sampling [--image path] [--samples N] "
                                            "[--iterations N] [--indirect]");
        }
        if (!count || !iterations)
            throw std::invalid_argument("samples and iterations must be positive");
        auto image = metalrt_examples::load_image(path);
        auto device = Device::system_default();
        const auto upload_start = Clock::now();
        auto texture = device.upload_texture({image.width, image.height, PixelFormat::RGBA8Unorm},
                                             image.rgba.data(), image.rgba.size());
        const double upload = seconds(upload_start);
        auto sampler = device.sampler();
        auto uv = device.allocate(checked_mul(count, sizeof(UV)));
        std::mt19937 rng{12345};
        for (size_t i = 0; i < count; ++i)
            uv.data<UV>()[i] = {float(rng() >> 8) * 0x1p-24f, float(rng() >> 8) * 0x1p-24f};
        auto output = device.allocate(checked_mul(count, 16));
        auto baseline = device.allocate(output.size());
        auto pixels = device.allocate(image.rgba.size());
        std::memcpy(pixels.contents(), image.rgba.data(), pixels.size());
        const std::string shader =
            std::string{METALRT_EXAMPLE_SHADER_DIR} + "/texture_sampling.metal";
        auto hardware = device.kernel_file(shader, "sample_image");
        auto manual = device.kernel_file(shader, "sample_image_buffer");
        Stream stream{device, policy};
        Graph sampled, buffered;
        {
            auto record = sampled.record_guard(&stream);
            launch_linear(&stream, hardware, count,
                          {Argument::texture(0, texture), Argument::sampler(0, sampler),
                           Argument::buffer(0, uv.view()),
                           Argument::buffer(1, output.view(), Access::Write),
                           Argument::scalar(2, count)});
            record.finish();
        }
        {
            auto record = buffered.record_guard(&stream);
            launch_linear(&stream, manual, count,
                          {Argument::buffer(0, pixels.view()), Argument::buffer(1, uv.view()),
                           Argument::buffer(2, baseline.view(), Access::Write),
                           Argument::scalar(3, Dimensions{uint32_t(image.width),
                                                          uint32_t(image.height), count})});
            record.finish();
        }
        sampled.prepare(&stream);
        buffered.prepare(&stream);
        for (int i = 0; i < 3; ++i) {
            sampled.launch(&stream);
            buffered.launch(&stream);
        }
        stream.synchronize();
        float maximum_error = 0;
        for (size_t i = 0; i < size_t(count) * 4; ++i) {
            if (!std::isfinite(output.data<float>()[i]) ||
                !std::isfinite(baseline.data<float>()[i]))
                throw std::runtime_error("non-finite GPU sample");
            maximum_error = std::max(maximum_error,
                                     std::abs(output.data<float>()[i] - baseline.data<float>()[i]));
        }
        if (!std::isfinite(maximum_error) || maximum_error > .006f)
            throw std::runtime_error("hardware bilinear samples disagree with buffer baseline");
        // A small independent CPU check, including each texel's normalized RGBA conversion.
        auto cpu = Device::reference();
        auto expected =
            cpu.upload_texture(texture.descriptor(), image.rgba.data(), image.rgba.size());
        auto cpu_sampler = cpu.sampler();
        for (size_t i = 0; i < std::min<size_t>(count, 4096); ++i) {
            const auto xy = uv.data<UV>()[i];
            auto value = expected.reference_sample(cpu_sampler, xy.u, xy.v);
            for (size_t k = 0; k < 4; ++k)
                if (std::abs(value[k] - output.data<float>()[4 * i + k]) > .006f)
                    throw std::runtime_error("GPU sample disagrees with CPU reference");
        }
        auto benchmark = [&](Graph& graph, const char* label) {
            stream.begin_gpu_timing();
            const auto start = Clock::now();
            for (uint32_t i = 0; i < iterations; ++i)
                graph.launch(&stream);
            const auto timing = stream.end_gpu_timing();
            const double wall = seconds(start);
            if (!timing.valid())
                throw std::runtime_error("GPU timestamps unavailable");
            const double gpu = timing.command_buffer_seconds;
            std::cout << label << ": " << gpu * 1e3 / iterations << " ms/batch, "
                      << double(count) * iterations / gpu / 1e6
                      << " M samples/s (GPU command-buffer time); " << wall * 1e3 / iterations
                      << " ms/batch wall\n";
            return gpu;
        };
        std::cout << std::fixed << std::setprecision(3) << "GPU: " << device.capabilities().name
                  << "\nImage: " << path << " (" << image.width << 'x' << image.height << ")\n"
                  << "Private texture upload: " << upload * 1e3 << " ms, " << image.rgba.size()
                  << " bytes\n"
                  << "Samples/batch: " << count << ", iterations: " << iterations
                  << "; maximum hardware/buffer difference: " << maximum_error << '\n';
        auto buffer_time = benchmark(buffered, "Manual bilinear buffer");
        auto texture_time = benchmark(sampled, "Hardware bilinear texture");
        std::cout << "Texture speedup on this workload: " << buffer_time / texture_time << "x\n"
                  << "Verified GPU samples; output is a retained float4 GPU buffer.\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
