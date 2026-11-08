// Execution contracts across supported backends and dispatch policies.
#include "support.h"
namespace runtime_tests {
void policy_sequence(Device d, const Kernels& k, DispatchPolicy policy) {
    constexpr uint32_t n = 257; // Deliberately not a threadgroup multiple.
    Stream stream(d, policy);
    auto input = d.allocate(n * sizeof(float)), intermediate = d.allocate(n * sizeof(float));
    auto output = d.allocate(n * sizeof(float)), replacement = d.allocate(n * sizeof(float));
    for (uint32_t i = 0; i < n; ++i) {
        input.data<float>()[i] = float(i) / 32;
        replacement.data<float>()[i] = float(i) / 64 + 2;
    }
    Graph graph;
    auto capture = [&](BufferView in, float scale, float bias) {
        auto scope = graph.capture_guard(&stream);
        check(stream.is_capturing(), "capture flag must be set inside the guard");
        affine(stream, k.affine, std::move(in), intermediate.view(), n, scale, bias);
        square(stream, k.square, intermediate.view(), output.view(), n);
        scope.finish();
    };
    capture(input.view(), 2, 1);
    stream.synchronize();
    check(!stream.is_recording(), "capture must end after finish");
    auto stats = stream.statistics();
    check(stats.compute_dispatches == 2, "two dependent compute dispatches expected");
    check(stats.command_buffers == (policy == DispatchPolicy::PerKernel ? 2u : 1u),
          "policy command-buffer count mismatch");
    check(stats.blit_commands == 0, "zero-copy kernel chain must not submit a transfer");
    for (uint32_t i = 0; i < n; ++i)
        near(output.data<float>()[i], std::pow(2 * input.data<float>()[i] + 1, 2),
             "dependent chain");
    const auto commands = graph.commands();
    check(commands.size() == 2, "graph stores both dispatches");
    check(commands[0].compute.argument(1).resource == commands[1].compute.argument(0).resource,
          "writer and reader must share the exact intermediate view");
    graph.launch(&stream).wait();
    stream.synchronize();
    if (policy == DispatchPolicy::Indirect) {
        stats = stream.statistics();
        check(stats.icb_builds == 1 && stats.icb_reuses == 1,
              "identical replay must reuse its ICB plan");
    }
    capture(input.view(), -0.5f, 4);
    stream.synchronize();
    for (uint32_t i = 0; i < n; ++i)
        near(output.data<float>()[i], std::pow(-0.5f * input.data<float>()[i] + 4, 2),
             "updated scalar snapshot");
    capture(replacement.view(), 0.75f, -1);
    stream.synchronize();
    for (uint32_t i = 0; i < n; ++i)
        near(output.data<float>()[i], std::pow(0.75f * replacement.data<float>()[i] - 1, 2),
             "updated buffer binding");
    if (policy == DispatchPolicy::Indirect) {
        stats = stream.statistics();
        check(stats.icb_updates + stats.icb_builds >= 3,
              "changed bindings must update or version the executable plan");
    }
}

void temporary_lifetimes(Device d, const Kernels& k, DispatchPolicy policy) {
    constexpr uint32_t n = 19;
    Stream stream(d, policy);
    auto output = d.allocate(n * sizeof(float));
    Graph graph;
    {
        auto input = d.allocate(n * sizeof(float));
        auto temporary = d.allocate(n * sizeof(float));
        for (uint32_t i = 0; i < n; ++i)
            input.data<float>()[i] = float(i);
        auto guard = graph.capture_guard(&stream);
        affine(stream, k.affine, input.view(), temporary.view(), n, 2, 1);
        square(stream, k.square, temporary.view(), output.view(), n);
    } // Graph/submission must retain input and temporary after both handles die.
    auto ticket = graph.launch(&stream);
    graph.reset(); // In-flight commands must retain their own resources and plan.
    ticket.wait();
    stream.synchronize();
    for (uint32_t i = 0; i < n; ++i)
        near(output.data<float>()[i], std::pow(2 * float(i) + 1, 2), "retained temporary result");
    check(graph.size() == 0, "reset clears recorded commands");
}

void private_memory_and_byte_transfers(Device d, const Kernels& k, DispatchPolicy policy) {
    Stream stream(d, policy);
    constexpr uint32_t n = 17;
    auto input = d.allocate(n * sizeof(float)),
         private_buffer = d.allocate(n * sizeof(float), StorageMode::Private);
    auto output = d.allocate(n * sizeof(float));
    for (uint32_t i = 0; i < n; ++i)
        input.data<float>()[i] = float(i);
    throws([&] { private_buffer.contents(); }, "private memory must not expose a CPU pointer");
    {
        auto batch = stream.batch();
        stream.copy(input.view(), private_buffer.view());
        affine(stream, k.affine, private_buffer.view(), output.view(), n, 2, -1);
    }
    stream.synchronize();
    for (uint32_t i = 0; i < n; ++i)
        near(output.data<float>()[i], 2 * float(i) - 1, "private memory compute");
    auto bytes = d.allocate(32, StorageMode::Private), readback = d.allocate(32);
    {
        auto batch = stream.batch();
        stream.fill(bytes.view(), 0);
        stream.fill(bytes.view(3, 17), 0xa5); // Tests the native unaligned byte fallback.
        stream.copy(bytes.view(1, 23), readback.view(5, 23));
    }
    stream.synchronize();
    auto p = readback.data<uint8_t>();
    for (size_t i = 0; i < 23; ++i)
        check(p[5 + i] == ((i + 1 >= 3 && i + 1 < 20) ? 0xa5 : 0),
              "unaligned private copy/fill result");
    auto shifted_input = d.allocate(32 * sizeof(float)),
         shifted_output = d.allocate(32 * sizeof(float));
    for (size_t i = 0; i < 32; ++i) {
        shifted_input.data<float>()[i] = float(i);
        shifted_output.data<float>()[i] = -123;
    }
    affine(stream, k.affine, shifted_input.view(3 * sizeof(float), 9 * sizeof(float)),
           shifted_output.view(5 * sizeof(float), 9 * sizeof(float)), 9, 2, 1);
    stream.synchronize();
    for (size_t i = 0; i < 32; ++i)
        near(shifted_output.data<float>()[i], i >= 5 && i < 14 ? 2 * float(i - 2) + 1 : -123,
             "nonzero buffer binding offsets and untouched guards");
}

void dynamic_threadgroup_memory(Device d, const Kernels& k, DispatchPolicy policy) {
    constexpr uint32_t n = 257;
    Stream stream(d, policy);
    auto input = d.allocate(n * sizeof(float)), output = d.allocate(n * sizeof(float));
    for (uint32_t i = 0; i < n; ++i)
        input.data<float>()[i] = float(i) + 0.5f;
    Graph graph;
    auto launch = [&] {
        auto capture = graph.capture_guard(&stream);
        stream.dispatch(k.threadgroup_exchange, {n, 1, 1}, {64, 1, 1},
                        {Argument::buffer(0, input.view()),
                         Argument::buffer(1, output.view(), Access::Write), Argument::scalar(2, n),
                         Argument::threadgroup(0, 64 * sizeof(float))});
        capture.finish();
        stream.synchronize();
    };
    try {
        launch();
    } catch (const std::runtime_error& e) {
        const std::string prefix = "Indirect dynamic threadgroup memory requires macOS 14";
        if (policy == DispatchPolicy::Indirect && d.capabilities().backend == BackendKind::Metal &&
            std::string(e.what()).compare(0, prefix.size(), prefix) == 0) {
            check(!stream.is_recording(),
                  "unsupported indirect threadgroup memory must cancel capture cleanly");
            check(stream.statistics().command_buffers == 0,
                  "unsupported indirect threadgroup memory must fail before submission");
            throw TestSkipped{std::string("native indirect dynamic threadgroup kernel: ") +
                              e.what()};
        }
        throw;
    }
    for (uint32_t i = 0; i < n; ++i)
        near(output.data<float>()[i], (i ^ 1u) < n ? input.data<float>()[i ^ 1u] : 0.0f,
             "dynamic threadgroup exchange");
    graph.launch(&stream).wait();
    stream.synchronize();
    for (uint32_t i = 0; i < n; ++i)
        near(output.data<float>()[i], (i ^ 1u) < n ? input.data<float>()[i ^ 1u] : 0.0f,
             "replayed threadgroup exchange");
}

} // namespace runtime_tests
