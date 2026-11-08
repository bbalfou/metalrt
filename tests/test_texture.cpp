#include <metalrt/runtime.h>
#include <metalrt/detail/backend.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>

using namespace metalrt;
namespace {
void check(bool condition, const char* message) {
    if (!condition)
        throw std::runtime_error(message);
}
size_t rejection_case = 0;
template <class F> void rejects(F&& f) {
    ++rejection_case;
    try {
        f();
    } catch (const std::invalid_argument&) {
        return;
    }
    throw std::runtime_error("invalid texture operation was accepted (case " +
                             std::to_string(rejection_case) + ")");
}
void near(float a, float b, float tolerance = .003f) {
    if (!std::isfinite(a) || std::abs(a - b) > tolerance)
        throw std::runtime_error("sample mismatch: " + std::to_string(a) + " vs " +
                                 std::to_string(b));
}
struct UV {
    float u, v;
};
const std::vector<UV> coordinates{
    {.25f, .25f},   {.75f, .25f}, {.25f, .75f},  {.75f, .75f},   {.5f, .5f}, {0, 0},  {1, 1},
    {-.25f, 1.25f}, {.1f, .8f},   {1.7f, -2.1f}, {.501f, .499f}, {0, .5f},   {1, .5f}};
Kernel sample_kernel(Device device) {
    return device.kernel_file(
        std::string{METALRT_TEST_SHADER_DIR} + "/texture_sampling.metal", "sample_image",
        [](const ComputeCommand& c) {
            const auto* uv = c.buffer<UV>(0);
            auto* rgb = c.buffer<std::array<float, 4>>(1);
            for (size_t i = 0; i < std::min<size_t>(c.scalar<uint32_t>(2), c.grid.x); ++i)
                rgb[i] = c.texture(0).reference_sample(c.sampler(0), uv[i].u, uv[i].v);
        });
}
std::vector<Argument> bindings(Texture2D texture, Sampler sampler, Buffer uv, Buffer output) {
    return {Argument::texture(0, texture), Argument::sampler(0, sampler),
            Argument::buffer(0, uv.view()), Argument::buffer(1, output.view(), Access::Write),
            Argument::scalar(2, uint32_t(coordinates.size()))};
}
void exercise(Device device, DispatchPolicy policy) {
    const std::array<uint8_t, 16> pixels{255, 0, 0,   255, 0,   255, 0,   255,
                                         0,   0, 255, 255, 255, 255, 255, 255};
    auto texture =
        device.upload_texture({2, 2, PixelFormat::RGBA8Unorm}, pixels.data(), pixels.size());
    auto cpu = Device::reference();
    auto expected =
        cpu.upload_texture({2, 2, PixelFormat::RGBA8Unorm}, pixels.data(), pixels.size());
    auto uv = device.allocate(coordinates.size() * sizeof(UV));
    std::memcpy(uv.contents(), coordinates.data(), uv.size());
    auto output = device.allocate(coordinates.size() * 4 * sizeof(float));
    auto sums = device.allocate(coordinates.size() * sizeof(float));
    const auto sample = sample_kernel(device);
    auto consume =
        device.kernel_file(std::string{METALRT_TEST_SHADER_DIR} + "/texture_sampling.metal",
                           "consume_samples", [](const ComputeCommand& c) {
                               for (uint32_t i = 0; i < c.scalar<uint32_t>(2); ++i) {
                                   const auto* v = c.buffer<float>(0) + 4 * i;
                                   c.buffer<float>(1)[i] = v[0] + v[1] + v[2] + v[3];
                               }
                           });
    Stream stream{device, policy};
    stream.fill(output.view());
    Graph graph;
    auto sampler = device.sampler();
    {
        auto recording = graph.record_guard(&stream);
        // Buffer-only -> texture -> buffer-only also checks hybrid ICB segment boundaries.
        launch_linear(&stream, consume, coordinates.size(),
                      {Argument::buffer(0, output.view()),
                       Argument::buffer(1, sums.view(), Access::Write),
                       Argument::scalar(2, uint32_t(coordinates.size()))});
        launch_linear(&stream, sample, coordinates.size(), bindings(texture, sampler, uv, output));
        launch_linear(&stream, consume, coordinates.size(),
                      {Argument::buffer(0, output.view()),
                       Argument::buffer(1, sums.view(), Access::Write),
                       Argument::scalar(2, uint32_t(coordinates.size()))});
        recording.finish();
    }
    graph.prepare(&stream);
    for (auto filter : {Filter::Nearest, Filter::Linear})
        for (auto address : {AddressMode::ClampToEdge, AddressMode::Repeat})
            for (auto address_v : {AddressMode::ClampToEdge, AddressMode::Repeat}) {
                SamplerDescriptor desc{filter, address, address_v};
                auto update = graph.update();
                update.set_sampler(1, 0, device.sampler(desc)).commit();
                graph.launch(&stream).wait();
                graph.launch(&stream).wait();
                auto cpu_sampler = cpu.sampler(desc);
                for (size_t i = 0; i < coordinates.size(); ++i) {
                    const auto value =
                        expected.reference_sample(cpu_sampler, coordinates[i].u, coordinates[i].v);
                    float sum = 0;
                    for (size_t k = 0; k < 4; ++k) {
                        near(output.data<float>()[4 * i + k], value[k]);
                        sum += value[k];
                    }
                    near(sums.data<float>()[i], sum, .01f);
                }
            }
    if (policy == DispatchPolicy::Indirect)
        check(stream.statistics().icb_reuses > 0, "buffer-only segments were not reused");

    // Submit old and new immutable versions without a host wait between them.
    auto previous_output = device.allocate(output.size());
    {
        auto update = graph.update();
        update.set_buffer(1, 1, previous_output.view()).commit();
    }
    const std::array<uint8_t, 4> replacement{64, 128, 192, 255};
    auto newer = device.upload_texture({1, 1, PixelFormat::RGBA8Unorm}, replacement.data(),
                                       replacement.size());
    std::weak_ptr<detail::TextureAllocation> retained = newer.implementation();
    auto old = graph.launch(&stream);
    {
        auto update = graph.update();
        update.set_texture(1, 0, newer).set_buffer(1, 1, output.view()).commit();
    }
    newer = {};
    check(!retained.expired(), "graph failed to retain its texture");
    graph.launch(&stream).wait();
    old.wait();
    near(output.data<float>()[0], 64.f / 255);
    check(std::abs(previous_output.data<float>()[0] - output.data<float>()[0]) > .01f,
          "graph texture update changed an older submission");
    graph = Graph{};
    stream.synchronize();

    // GPU-produced threadgroup counts must still work with texture/sampler arguments.
    auto groups = device.allocate(3 * sizeof(uint32_t));
    groups.data<uint32_t>()[0] = 1;
    groups.data<uint32_t>()[1] = 1;
    groups.data<uint32_t>()[2] = 1;
    stream.dispatch_indirect(sample, groups.view(), {1, 1, 1}, {32, 1, 1},
                             bindings(texture, sampler, uv, output));
    stream.synchronize();
    near(output.data<float>()[0], 1);
}
void formats_and_stride(Device device) {
    for (auto format : {PixelFormat::R8Unorm, PixelFormat::RGBA8Unorm, PixelFormat::R32Float,
                        PixelFormat::RGBA32Float}) {
        const size_t row = 3 * pixel_size(format), stride = row + 7;
        std::vector<uint8_t> pixels(stride + row, 0xCD);
        const bool floating = format == PixelFormat::R32Float || format == PixelFormat::RGBA32Float;
        const bool rgba = format == PixelFormat::RGBA8Unorm || format == PixelFormat::RGBA32Float;
        for (size_t y = 0; y < 2; ++y)
            for (size_t x = 0; x < 3; ++x)
                for (size_t k = 0; k < (rgba ? 4u : 1u); ++k) {
                    const auto offset =
                        y * stride + x * pixel_size(format) + k * (floating ? 4 : 1);
                    if (floating) {
                        float value = float(x + 3 * y + k) / 10;
                        std::memcpy(pixels.data() + offset, &value, 4);
                    } else
                        pixels[offset] = uint8_t(20 * (x + 3 * y + k));
                }
        auto texture = device.upload_texture({3, 2, format}, pixels.data(), pixels.size(), stride);
        auto cpu = Device::reference();
        auto expected = cpu.upload_texture({3, 2, format}, pixels.data(), pixels.size(), stride);
        std::fill(pixels.begin(), pixels.end(), 0); // upload must copy, never borrow host memory
        auto uv = device.allocate(coordinates.size() * sizeof(UV));
        std::memcpy(uv.contents(), coordinates.data(), uv.size());
        auto output = device.allocate(coordinates.size() * 4 * sizeof(float));
        Stream stream{device};
        launch_linear(&stream, sample_kernel(device), coordinates.size(),
                      bindings(texture, device.sampler(), uv, output));
        stream.synchronize();
        auto sampler = cpu.sampler();
        for (size_t i = 0; i < coordinates.size(); ++i) {
            const auto value =
                expected.reference_sample(sampler, coordinates[i].u, coordinates[i].v);
            for (size_t k = 0; k < 4; ++k)
                near(output.data<float>()[4 * i + k], value[k]);
        }
    }
}
void namespaces_and_lifetimes(Device device) {
    const uint8_t pixels[]{255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255, 255, 255, 255, 255};
    auto uv = device.allocate(sizeof(UV));
    uv.data<UV>()[0] = {.5f, .5f};
    auto output = device.allocate(16);
    Stream stream{device};
    Graph graph;
    std::weak_ptr<detail::TextureAllocation> weak_texture;
    std::weak_ptr<detail::SamplerState> weak_sampler;
    {
        auto texture = device.upload_texture({2, 2}, pixels, sizeof(pixels));
        auto sampler = device.sampler();
        weak_texture = texture.implementation();
        weak_sampler = sampler.implementation();
        Command command;
        command.compute.kernel = sample_kernel(device);
        command.compute.grid = {1, 1, 1};
        command.compute.group = {32, 1, 1};
        command.compute.arguments = {
            Argument::texture(0, texture),   Argument::sampler(0, sampler),
            Argument::texture(3, texture),   Argument::sampler(3, sampler),
            Argument::texture(4, texture),   Argument::sampler(4, sampler),
            Argument::buffer(0, uv.view()),  Argument::buffer(1, output.view(), Access::Write),
            Argument::scalar(2, uint32_t{1})};
        command.compute.optional_buffer_mask = 3;
        command.compute.pointer_base_arguments = {{0, 4}};
        auto materialized = detail::materialize_compute_arguments(command.compute);
        check(std::count_if(materialized.begin(), materialized.end(),
                            [](const Argument& a) { return a.index == 4; }) == 3,
              "pointer base confused texture/sampler and buffer namespaces");
        auto record = graph.record_guard(&stream);
        stream.enqueue(std::move(command));
        record.finish();
    }
    check(!weak_texture.expired() && !weak_sampler.expired(), "recording did not retain resources");
    auto ticket = graph.launch(&stream);
    graph.reset(); // Drop every application texture/sampler handle before waiting.
    ticket.wait();
    // Independent analytic value for the center of a red/green/blue/white image.
    near(output.data<float>()[0], .5f);
    near(output.data<float>()[1], .5f);
    near(output.data<float>()[2], .5f);
    near(output.data<float>()[3], 1.f);
    stream.synchronize();
}
void validation(Device device) {
    uint8_t pixels[16]{};
    rejects([&] { device.upload_texture({0, 2}, pixels, sizeof(pixels)); });
    rejects([&] { device.upload_texture({16385, 1}, pixels, sizeof(pixels)); });
    rejects([&] { device.upload_texture({2, 2}, nullptr, sizeof(pixels)); });
    rejects([&] { device.upload_texture({2, 2}, pixels, 15); });
    rejects([&] { device.upload_texture({2, 2}, pixels, 16, 7); });
    rejects([&] { Argument::texture(0, {}); });
    rejects([&] { Argument::sampler(0, {}); });
    rejects([&] { device.sampler({static_cast<Filter>(99)}); });
    rejects([&] { device.sampler({Filter::Linear, static_cast<AddressMode>(99)}); });
    rejects([&] { device.upload_texture({2, 2, static_cast<PixelFormat>(99)}, pixels, 16); });
    auto texture = device.upload_texture({2, 2}, pixels, sizeof(pixels));
    auto sampler = device.sampler();
    auto uv = device.allocate(coordinates.size() * sizeof(UV));
    auto output = device.allocate(coordinates.size() * 16);
    Stream stream{device};
    const auto kernel = sample_kernel(device);
    auto args = bindings(texture, sampler, uv, output);
    args.push_back(Argument::texture(0, texture));
    rejects([&] { launch_linear(&stream, kernel, 1, args); });
    args = bindings(texture, sampler, uv, output);
    args.push_back(Argument::sampler(0, sampler));
    rejects([&] { launch_linear(&stream, kernel, 1, args); });
    args = bindings(texture, sampler, uv, output);
    args[0].index = 31;
    rejects([&] { launch_linear(&stream, kernel, 1, args); });
    args = bindings(texture, sampler, uv, output);
    args[1].index = 16;
    rejects([&] { launch_linear(&stream, kernel, 1, args); });
    Device other{detail::make_reference_device()};
    args = bindings(other.upload_texture({2, 2}, pixels, sizeof(pixels)), sampler, uv, output);
    rejects([&] { launch_linear(&stream, kernel, 1, args); });
    args = bindings(texture, other.sampler(), uv, output);
    rejects([&] { launch_linear(&stream, kernel, 1, args); });
    Graph graph;
    {
        auto recording = graph.record_guard(&stream);
        launch_linear(&stream, kernel, 1, bindings(texture, sampler, uv, output));
        recording.finish();
    }
    auto invalid = graph.update();
    invalid.set_texture(0, 0, {});
    rejects([&] { invalid.commit(); });
    check(graph.revision() == 1, "failed update changed graph");
    check(device.statistics().texture_allocations > 0, "missing texture statistics");
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
        validation(device);
        formats_and_stride(device);
        namespaces_and_lifetimes(device);
        for (auto policy :
             {DispatchPolicy::PerKernel, DispatchPolicy::Direct, DispatchPolicy::Indirect})
            if (policy != DispatchPolicy::Indirect || device.capabilities().indirect_compute)
                exercise(device, policy);
        std::cout << "PASS texture upload, sampling, formats, stride, replay, updates, lifetimes, "
                     "dynamic dispatch on "
                  << device.capabilities().name << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "TEXTURE TEST FAILURE: " << error.what() << '\n';
        return 1;
    }
}
