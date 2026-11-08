// Optional buffers contracts across supported backends and dispatch policies.
#include "support.h"
namespace runtime_tests {
const char* optional_source = R"MSL(
#include <metal_stdlib>
using namespace metal;
struct OptionalPayload { float values[64]; };
kernel void optional_inputs(device const float* input [[buffer(0)]],device float* output [[buffer(1)]],
                            device const OptionalPayload* bias [[buffer(2)]],constant uint& n [[buffer(3)]],
                            constant uint& present [[buffer(4)]],
                            uint i [[thread_position_in_grid]]) {
    input=(present&1u)?input:nullptr;
    output=(present&2u)?output:nullptr;
    bias=(present&4u)?bias:nullptr;
    if(i<n && output)output[i]=(input?input[i]:-2.0f)+(bias?bias->values[63]:0.0f);
})MSL";

void nullable_buffers(Device d, DispatchPolicy policy) {
    constexpr uint32_t n = 7;
    auto kernel = d.kernel(optional_source, "optional_inputs", [](const ComputeCommand& command) {
        const auto n = command.scalar<uint32_t>(3);
        const auto input = command.buffer<float>(0), bias = command.buffer<float>(2);
        auto output = command.buffer<float>(1);
        check(command.scalar<uint32_t>(4) ==
                  ((input ? 1u : 0u) | (output ? 2u : 0u) | (bias ? 4u : 0u)),
              "derived mask must describe semantic buffers in Reference callbacks");
        if (output)
            for (uint32_t i = 0; i < n; ++i)
                output[i] = (input ? input[i] : -2) + (bias ? bias[63] : 0);
    });
    Stream stream(d, policy);
    // The optional struct exceeds the fallback floor. Its native reflected
    // size must be honored even though no struct bytes are read when absent.
    auto input = d.allocate(n * sizeof(float)), output = d.allocate(n * sizeof(float)),
         bias = d.allocate(64 * sizeof(float));
    for (uint32_t i = 0; i < n; ++i)
        input.data<float>()[i] = float(i);
    bias.data<float>()[63] = 3;
    Command command;
    command.compute.kernel = kernel;
    command.compute.grid = {n, 1, 1};
    command.compute.group = {32, 1, 1};
    command.compute.optional_buffer_mask = 4;
    command.compute.arguments = {Argument::null_buffer(0),
                                 Argument::buffer(1, output.view(), Access::Write),
                                 Argument::null_buffer(2), Argument::scalar(3, n)};
    Graph graph;
    {
        auto recording = graph.record_guard(&stream);
        stream.enqueue(command);
    }
    graph.prepare(&stream);
    const auto prepared_allocations = d.statistics().buffer_allocations;
    graph.launch(&stream).wait();
    graph.launch(&stream).wait();
    check(d.statistics().buffer_allocations == prepared_allocations,
          "replaying prepared optional bindings must reuse fallback buffers and mask constants");
    for (uint32_t i = 0; i < n; ++i)
        near(output.data<float>()[i], -2, "optional absent input");
    graph.update().set_buffer(0, 0, input.view()).set_buffer(0, 2, bias.view()).commit();
    graph.launch(&stream).wait();
    for (uint32_t i = 0; i < n; ++i)
        near(output.data<float>()[i], float(i) + 3, "null buffer rebound in graph");
    graph.update().set_null_buffer(0, 0).set_null_buffer(0, 2).commit();
    graph.launch(&stream).wait();
    for (uint32_t i = 0; i < n; ++i)
        near(output.data<float>()[i], -2, "optional argument reset during graph update");
    graph.update().set_null_buffer(0, 1).commit();
    for (uint32_t i = 0; i < n; ++i)
        output.data<float>()[i] = 91;
    graph.launch(&stream).wait();
    for (uint32_t i = 0; i < n; ++i)
        near(output.data<float>()[i], 91, "absent output must remain semantically absent");
    graph.update().set_buffer(0, 1, output.view()).commit();
    graph.launch(&stream).wait();
    for (uint32_t i = 0; i < n; ++i)
        near(output.data<float>()[i], -2, "restored output must use a fresh derived mask");
    throws([&] { graph.update().set_scalar(0, 4, uint32_t{7}).commit(); },
           "hidden presence masks must not be graph-overridable");

    const auto snapshot = graph.commands();
    auto changed = snapshot;
    changed[0].compute.optional_buffer_mask = 5;
    check(!detail::same_commands(snapshot, changed),
          "optional mask metadata must participate in executable identity");
    changed = snapshot;
    changed[0].compute.pointer_base_arguments = {{0, 6}};
    check(!detail::same_commands(snapshot, changed),
          "pointer-base metadata must participate in executable identity");
    const auto rejects_command = [&](Command invalid) {
        throws([&] { stream.enqueue(std::move(invalid)); },
               "invalid derived kernel binding must fail before submission");
    };
    auto invalid = command;
    invalid.compute.optional_buffer_mask = 31;
    rejects_command(invalid);
    invalid = command;
    invalid.compute.optional_buffer_mask = 3;
    rejects_command(invalid);
    invalid = command;
    invalid.compute.arguments.push_back(Argument::scalar(4, uint32_t{0}));
    rejects_command(invalid);
    invalid = command;
    invalid.compute.pointer_base_arguments = {{0, 4}};
    rejects_command(invalid);
    invalid = command;
    invalid.compute.pointer_base_arguments = {{4, 6}};
    rejects_command(invalid);
    invalid = command;
    invalid.compute.pointer_base_arguments = {{0, 6}, {0, 6}};
    rejects_command(invalid);
    invalid = command;
    invalid.compute.pointer_base_arguments = {{0, 6}, {6, 7}};
    rejects_command(invalid);
    invalid = command;
    invalid.compute.pointer_base_arguments = {{3, 6}};
    rejects_command(invalid);
    invalid = command;
    invalid.compute.pointer_base_arguments = {{5, 6}};
    rejects_command(invalid);
    throws([&] { Argument::buffer(0, {}); },
           "ordinary empty buffer binding must still be rejected");
    auto malformed = Argument::null_buffer(0);
    malformed.resource = input.view();
    throws([&] { stream.dispatch(kernel, {n, 1, 1}, {32, 1, 1}, {malformed}); },
           "null buffer cannot hide a resource");

    auto unmasked = command;
    unmasked.compute.optional_buffer_mask = UINT32_MAX;
    unmasked.compute.arguments.push_back(Argument::scalar(4, uint32_t{2}));
    if (d.capabilities().backend == BackendKind::Metal) {
        bool precise = false;
        try {
            stream.enqueue(unmasked);
        } catch (const std::invalid_argument& error) {
            precise = std::string(error.what()).find("optional_buffer_mask") != std::string::npos;
        }
        check(precise, "active native null binding must fail clearly before Metal API validation");
    } else {
        stream.enqueue(unmasked);
        stream.synchronize();
        near(output.data<float>()[0], -2,
             "low-level Reference null binding remains a real nullptr");
    }
    const std::string unused_source = R"MSL(
#include <metal_stdlib>
using namespace metal;
kernel void ignored_buffer(const device float* unused [[buffer(0)]],device float* output [[buffer(1)]]) {
    output[0]=17.0f;
})MSL";
    auto unused = d.kernel(unused_source, "ignored_buffer",
                           [](const ComputeCommand& c) { c.buffer<float>(1)[0] = 17; });
    stream.dispatch(unused, {1, 1, 1}, {1, 1, 1},
                    {Argument::null_buffer(0), Argument::buffer(1, output.view(), Access::Write)});
    stream.synchronize();
    near(output.data<float>()[0], 17,
         "compiled-out native null slots remain usable without a mask");
}

} // namespace runtime_tests
