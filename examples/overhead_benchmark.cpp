// Submission overhead only: explicit completion waits are outside the timed
// interval. Use ring_pipeline for an example with overlapping CPU/GPU work.
#include <metalrt/parameters.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <string>

#ifndef METALRT_EXAMPLE_SHADER_DIR
#define METALRT_EXAMPLE_SHADER_DIR "kernels"
#endif

using namespace metalrt;
namespace {
using Clock = std::chrono::steady_clock;
struct Args {
    uint32_t count;
    float scale, bias;
};
static_assert(sizeof(Args) == 12, "file_dispatch.metal constant ABI");

struct Options {
    bool reference = false;
    uint32_t iterations = 200, warmup = 4, count = 257;
    std::string policy = "all";
};
uint32_t number(const std::string& text, bool zero_allowed) {
    if (text.empty() ||
        std::any_of(text.begin(), text.end(), [](char c) { return c < '0' || c > '9'; }))
        throw std::invalid_argument("expected an unsigned integer, got: " + text);
    const auto value = std::stoull(text);
    if (value > std::numeric_limits<uint32_t>::max() || (!zero_allowed && value == 0))
        throw std::invalid_argument("benchmark number is out of range: " + text);
    return static_cast<uint32_t>(value);
}
const char* policy_name(DispatchPolicy policy) {
    switch (policy) {
    case DispatchPolicy::PerKernel:
        return "per-kernel";
    case DispatchPolicy::Direct:
        return "direct";
    case DispatchPolicy::Indirect:
        return "indirect";
    }
    throw std::logic_error("invalid dispatch policy");
}
enum class Mode { Replay, Parameters, Scalar, Buffer, ScalarBuffer, Recapture };
const char* mode_name(Mode mode) {
    switch (mode) {
    case Mode::Replay:
        return "replay";
    case Mode::Parameters:
        return "parameter-buffer update";
    case Mode::Scalar:
        return "scalar update + replay";
    case Mode::Buffer:
        return "buffer rebind + replay";
    case Mode::ScalarBuffer:
        return "scalar+buffer + replay";
    case Mode::Recapture:
        return "recapture scalar+buffer";
    }
    throw std::logic_error("invalid benchmark mode");
}
Args parameters_for(uint32_t count, uint32_t iteration) {
    return {count, 0.5f + 0.125f * float(iteration % 7), float(iteration % 5)};
}
bool changes_input(Mode mode) {
    return mode == Mode::Buffer || mode == Mode::ScalarBuffer || mode == Mode::Recapture;
}

struct Result {
    double host_us = 0;
    Statistics runtime;
    DeviceStatistics device;
    uint64_t prepared_builds = 0;
};

Result run_case(Device device, const Kernel& kernel, const Buffer& input_a, const Buffer& input_b,
                const Buffer& output, DispatchPolicy policy, Mode mode, const Options& options) {
    Stream stream{device, policy};
    Graph graph;
    const Args initial = parameters_for(options.count, 0);
    std::optional<ParameterBuffer<Args>> parameters;
    if (mode == Mode::Parameters)
        parameters.emplace(device, initial);

    // Recording and preparation happen once, before warm-up and measurement.
    // Only the recapture case repeats this host-side command construction.
    auto record = [&](BufferView input, const Args& value, bool execute) {
        auto recording = execute ? graph.capture_guard(&stream) : graph.record_guard(&stream);
        launch_linear(&stream, kernel, options.count,
                      {Argument::buffer(0, std::move(input)),
                       Argument::buffer(1, output.view(), Access::Write),
                       parameters ? parameters->bind(2) : Argument::scalar(2, value)});
        recording.finish();
    };
    record(input_a.view(), initial, false);
    graph.prepare(&stream);

    auto submit = [&](uint32_t iteration) -> Ticket {
        switch (mode) {
        case Mode::Replay:
            return graph.launch(&stream);
        case Mode::Parameters:
            // First-class tracked host edit: the graph keeps the same buffer
            // identity and the same compiled command plan on every replay.
            return parameters->submit(stream, graph, parameters_for(options.count, iteration));
        case Mode::Scalar: {
            auto edit = graph.update();
            edit.set_scalar(0, 2, parameters_for(options.count, iteration));
            edit.commit();
            return graph.launch(&stream);
        }
        case Mode::Buffer: {
            auto edit = graph.update();
            edit.set_buffer(0, 0, (iteration % 2) ? input_b.view() : input_a.view());
            edit.commit();
            return graph.launch(&stream);
        }
        case Mode::ScalarBuffer: {
            auto edit = graph.update();
            edit.set_scalar(0, 2, parameters_for(options.count, iteration))
                .set_buffer(0, 0, (iteration % 2) ? input_b.view() : input_a.view());
            edit.commit();
            return graph.launch(&stream);
        }
        case Mode::Recapture:
            record((iteration % 2) ? input_b.view() : input_a.view(),
                   parameters_for(options.count, iteration), true);
            return stream.last_ticket();
        }
        throw std::logic_error("invalid benchmark mode");
    };

    for (uint32_t i = 0; i < options.warmup; ++i) {
        submit(i);
        stream.synchronize();
    }
    Result result;
    result.prepared_builds = stream.statistics().icb_builds;
    stream.reset_statistics();
    const auto before = device.statistics();
    uint64_t elapsed_ns = 0;
    for (uint32_t i = 0; i < options.iterations; ++i) {
        const auto begin = Clock::now();
        submit(i);
        elapsed_ns += static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - begin).count());
        // Deliberately one outstanding iteration. The GPU wait is separately
        // counted, but excluded from host submission timing. Shared parameter
        // edits therefore never race an earlier read, and plans are idle when
        // updated. This measures submission overhead, not streaming throughput.
        stream.synchronize();
    }
    result.host_us = double(elapsed_ns) / 1000.0 / options.iterations;
    result.runtime = stream.statistics();
    const auto after = device.statistics();
    result.device = {after.buffer_allocations - before.buffer_allocations,
                     after.allocated_bytes - before.allocated_bytes,
                     after.kernel_compilations - before.kernel_compilations};

    const uint32_t last = options.iterations - 1;
    const auto& input = changes_input(mode) && last % 2 ? input_b : input_a;
    const Args expected_args = (mode == Mode::Replay || mode == Mode::Buffer)
                                   ? initial
                                   : parameters_for(options.count, last);
    for (uint32_t i = 0; i < options.count; ++i) {
        const float expected = input.data<float>()[i] * expected_args.scale + expected_args.bias;
        const float actual = output.data<float>()[i];
        if (!std::isfinite(actual) ||
            std::abs(actual - expected) > 1e-5f * std::max(1.0f, std::abs(expected)))
            throw std::runtime_error(std::string(mode_name(mode)) +
                                     " produced an incorrect final result");
    }
    return result;
}

void print_row(Mode mode, const Result& result, uint32_t iterations) {
    const auto& s = result.runtime;
    const std::string icbs = std::to_string(s.icb_builds) + "/" + std::to_string(s.icb_updates) +
                             "/" + std::to_string(s.icb_reuses);
    const std::string allocations = std::to_string(result.device.buffer_allocations) + "/" +
                                    std::to_string(result.device.allocated_bytes);
    std::cout << std::left << std::setw(26) << mode_name(mode) << std::right << std::setw(11)
              << result.host_us << std::setw(11)
              << double(s.cpu_submit_nanoseconds) / 1000.0 / iterations << std::setw(11)
              << double(s.cpu_wait_nanoseconds) / 1000.0 / iterations << std::setw(9)
              << s.stream_wait_calls << std::setw(14) << icbs << std::setw(8)
              << result.prepared_builds << std::setw(14) << allocations << std::setw(10)
              << result.device.kernel_compilations << '\n';
}
} // namespace

int main(int argc, char** argv) {
    try {
        const char* usage =
            "usage: overhead_benchmark [--reference] [--policy all|per-kernel|direct|indirect] "
            "[--iterations N] [--warmup N] [--count N]";
        Options options;
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--reference")
                options.reference = true;
            else if (arg == "--help") {
                std::cout << usage << '\n';
                return 0;
            } else if (arg == "--policy" && i + 1 < argc)
                options.policy = argv[++i];
            else if (arg == "--iterations" && i + 1 < argc)
                options.iterations = number(argv[++i], false);
            else if (arg == "--warmup" && i + 1 < argc)
                options.warmup = number(argv[++i], true);
            else if (arg == "--count" && i + 1 < argc)
                options.count = number(argv[++i], false);
            else
                throw std::invalid_argument(usage);
        }
        if (options.policy != "all" && options.policy != "per-kernel" &&
            options.policy != "direct" && options.policy != "indirect")
            throw std::invalid_argument(usage);
        auto device = options.reference ? Device::reference() : Device::system_default();
        Device::set_current(device);
        const auto bytes = checked_mul(options.count, sizeof(float));
        auto input_a = device.allocate(bytes), input_b = device.allocate(bytes),
             output = device.allocate(bytes);
        for (uint32_t i = 0; i < options.count; ++i) {
            input_a.data<float>()[i] = float(i) / 1024.0f;
            input_b.data<float>()[i] = float(i) / 1024.0f + 2.0f;
        }
        // The identical MSL entry accepts a constant struct supplied either as
        // recorded scalar bytes or as a persistent parameter-buffer binding.
        auto kernel =
            device.kernel_file(std::string{METALRT_EXAMPLE_SHADER_DIR} + "/file_dispatch.metal",
                               "file_affine", [](const ComputeCommand& command) {
                                   const Args value =
                                       command.argument(2).kind == Argument::Kind::Bytes
                                           ? command.scalar<Args>(2)
                                           : *command.buffer<const Args>(2);
                                   const auto* input = command.buffer<const float>(0);
                                   auto* result = command.buffer<float>(1);
                                   for (uint32_t i = 0; i < value.count; ++i)
                                       result[i] = input[i] * value.scale + value.bias;
                               });
        std::cout << "Backend: " << device.capabilities().name << '\n';
        if (device.capabilities().backend == BackendKind::Reference)
            std::cout << "REFERENCE ONLY: these numbers describe CPU reference scheduling, not "
                         "Metal/GPU performance.\n";
        std::cout << options.iterations << " measured iterations, " << options.warmup
                  << " warm-up iterations, " << options.count << " elements per dispatch.\n"
                  << "Preparation/warm-up are excluded. Host us includes edits/recording and "
                     "submission; backend us is the queue-submit portion.\n"
                  << "One explicit completion wait per iteration is outside host timing. Wait us "
                     "is reported separately.\n"
                  << "ICBs are measured builds/updates/reuses; prep is builds before measurement. "
                     "Buffer allocs/bytes count runtime-created buffers,\n"
                  << "including native internal constant/scratch buffers, but excluding C++ heap, "
                     "ICB objects, and driver allocations.\n"
                  << "Compiles count public kernel compilations during measurement, excluding "
                     "backend-internal support pipelines.\n";
        std::cout << std::fixed << std::setprecision(3);
        for (auto policy :
             {DispatchPolicy::PerKernel, DispatchPolicy::Direct, DispatchPolicy::Indirect}) {
            if (options.policy != "all" && options.policy != policy_name(policy))
                continue;
            if (policy == DispatchPolicy::Indirect && !device.capabilities().indirect_compute) {
                std::cout << "SKIP indirect: compute ICBs unavailable on this device\n";
                if (options.policy == "indirect")
                    return 77;
                continue;
            }
            std::cout << "\nPolicy: " << policy_name(policy) << '\n'
                      << std::left << std::setw(26) << "Case" << std::right << std::setw(11)
                      << "host us" << std::setw(11) << "backend us" << std::setw(11) << "wait us"
                      << std::setw(9) << "waits" << std::setw(14) << "ICBs b/u/r" << std::setw(8)
                      << "prep" << std::setw(14) << "allocs/bytes" << std::setw(10) << "compiles"
                      << '\n';
            for (auto mode : {Mode::Replay, Mode::Parameters, Mode::Scalar, Mode::Buffer,
                              Mode::ScalarBuffer, Mode::Recapture})
                print_row(mode,
                          run_case(device, kernel, input_a, input_b, output, policy, mode, options),
                          options.iterations);
        }
        std::cout << "\nAll final outputs verified. No speedup or throughput guarantee is inferred "
                     "from this submission benchmark.\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
