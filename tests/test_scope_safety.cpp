#include <metalrt/control.h>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>

#ifndef METALRT_TEST_SHADER_DIR
#define METALRT_TEST_SHADER_DIR "kernels"
#endif

using namespace metalrt;
namespace {
void check(bool condition, const char* message) {
    if (!condition)
        throw std::runtime_error(message);
}
template <class F> void throws(F&& action, const char* message) {
    try {
        action();
    } catch (const std::exception&) {
        return;
    }
    throw std::runtime_error(message);
}
struct Affine {
    uint32_t n;
    float scale, bias;
};
struct Fixture {
    Device device;
    Stream stream;
    Kernel kernel;
    Buffer input, output, a, b, c, zero, one;
    Fixture(Device d, DispatchPolicy policy)
        : device(d), stream(d, policy),
          kernel(d.kernel_file(
              std::string{METALRT_TEST_SHADER_DIR} + "/runtime_tests.metal", "runtime_affine",
              [](const ComputeCommand& command) {
                  const auto p = command.scalar<Affine>(2);
                  for (size_t i = 0; i < std::min<size_t>(p.n, command.grid.x); ++i)
                      command.buffer<float>(1)[i] = p.scale * command.buffer<float>(0)[i] + p.bias;
              })),
          input(d.allocate(4)), output(d.allocate(4)), a(d.allocate(4)), b(d.allocate(4)),
          c(d.allocate(4)), zero(d.allocate(4)), one(d.allocate(4)) {
        input.data<float>()[0] = 1;
        output.data<float>()[0] = a.data<float>()[0] = b.data<float>()[0] = c.data<float>()[0] = 0;
        zero.data<uint32_t>()[0] = 0;
        one.data<uint32_t>()[0] = 1;
    }
    void emit(Buffer destination, float value) {
        stream.dispatch(kernel, {1, 1, 1}, {1, 1, 1},
                        {Argument::buffer(0, input.view()),
                         Argument::buffer(1, destination.view(), Access::Write),
                         Argument::scalar(2, Affine{1, 0, value})});
    }
    void execute(Graph& graph) {
        graph.launch(&stream).wait();
        stream.synchronize();
    }
    void expect(Buffer buffer, float value) {
        check(std::isfinite(buffer.data<float>()[0]) &&
                  std::abs(buffer.data<float>()[0] - value) < 1e-6f,
              "unexpected scope test output");
    }
};

void stale_scopes_after_batch_cancel(Fixture& f) {
    auto batch = f.stream.batch();
    auto parent = f.stream.predicate_scope(f.zero.view());
    auto child = f.stream.predicate_scope(f.one.view());
    f.emit(f.output, 99);
    batch.cancel();
    check(!f.stream.is_recording() && !f.stream.current_predicate(),
          "batch cancellation left nested state active");
    check(f.stream.statistics().command_buffers == 0, "cancelled batch submitted work");
    auto next = f.stream.batch();
    f.emit(f.output, 7);
    child.finish();
    parent.cancel(); // These invalidated tokens must not touch the new generation.
    check(f.stream.is_recording() && !f.stream.current_predicate(),
          "stale scope modified the next batch");
    next.finish();
    f.stream.synchronize();
    f.expect(f.output, 7);
}

void stale_scopes_after_capture_cancel(Fixture& f) {
    Graph cancelled;
    auto capture = cancelled.record_guard(&f.stream);
    auto parent = f.stream.predicate_scope(f.one.view());
    auto child = f.stream.predicate_scope(f.zero.view());
    f.emit(f.output, 99);
    capture.cancel();
    check(cancelled.size() == 0 && !f.stream.current_predicate(),
          "capture cancellation published commands or a predicate");
    Graph next;
    auto record = next.record_guard(&f.stream);
    f.emit(f.output, 8);
    parent.finish();
    child.cancel();
    record.finish();
    check(next.size() == 1 && !next.commands()[0].compute.predicate,
          "stale capture scope changed a later graph");
    f.execute(next);
    f.expect(f.output, 8);
}

void out_of_order_predicates(Fixture& f) {
    auto batch = f.stream.batch();
    f.emit(f.a, 1);
    auto parent = f.stream.predicate_scope(f.zero.view());
    f.emit(f.b, 99);
    auto child = f.stream.predicate_scope(f.one.view());
    f.emit(f.c, 99);
    throws([&] { parent.finish(); }, "out-of-order predicate finish was accepted");
    check(f.stream.is_recording() && !f.stream.current_predicate(),
          "predicate failure did not restore the enclosing batch");
    child.finish();
    f.emit(f.output, 9);
    batch.finish();
    f.stream.synchronize();
    f.expect(f.a, 1);
    f.expect(f.b, 0);
    f.expect(f.c, 0);
    f.expect(f.output, 9);
}

void out_of_order_transactions(Fixture& f) {
    Graph graph;
    auto record = graph.record_guard(&f.stream);
    f.emit(f.a, 1);
    auto parent = f.stream.transaction();
    f.emit(f.b, 99);
    auto child = f.stream.transaction();
    f.emit(f.c, 99);
    throws([&] { parent.finish(); }, "out-of-order transaction finish was accepted");
    child.finish();
    f.emit(f.output, 10);
    record.finish();
    check(graph.size() == 2, "transaction failure left its recorded suffix in the graph");
    f.execute(graph);
    f.expect(f.a, 1);
    f.expect(f.b, 0);
    f.expect(f.c, 0);
    f.expect(f.output, 10);
}

void live_transaction_blocks_parent_finish(Fixture& f) {
    auto batch = f.stream.batch();
    auto transaction = f.stream.transaction();
    f.emit(f.a, 99);
    throws([&] { batch.finish(); }, "batch finished while a transaction was live");
    transaction.finish();
    check(!f.stream.is_recording() && f.stream.statistics().command_buffers == 0,
          "failed parent finish submitted work");
    Graph graph;
    auto record = graph.record_guard(&f.stream);
    auto nested = f.stream.transaction();
    f.emit(f.b, 99);
    throws([&] { record.finish(); }, "capture finished while a transaction was live");
    nested.finish();
    check(graph.size() == 0 && !f.stream.is_recording(), "failed capture left graph/state active");
    f.emit(f.output, 11);
    f.stream.synchronize();
    f.expect(f.a, 0);
    f.expect(f.b, 0);
    f.expect(f.output, 11);
}

void cancel_restores_parallel_state(Fixture& f) {
    Graph graph;
    auto record = graph.record_guard(&f.stream);
    {
        auto parallel = f.stream.parallel();
        auto transaction = f.stream.transaction();
        f.emit(f.a, 99);
        transaction.cancel();
        f.emit(f.b, 12);
        parallel.finish();
    }
    {
        auto predicate = f.stream.predicate_scope(f.one.view());
        f.emit(f.c, 99);
        predicate.cancel();
    }
    Command next;
    next.compute.kernel = f.kernel;
    next.compute.grid = {1, 1, 1};
    next.compute.group = {1, 1, 1};
    next.compute.barrier_before = false;
    next.compute.arguments = {Argument::buffer(0, f.input.view()),
                              Argument::buffer(1, f.output.view(), Access::Write),
                              Argument::scalar(2, Affine{1, 0, 13})};
    f.stream.enqueue(std::move(next));
    record.finish();
    const auto commands = graph.commands();
    check(commands.size() == 2 && commands[0].compute.barrier_before &&
              commands[1].compute.barrier_before,
          "scope cancellation consumed parallel entry/join barriers");
    f.execute(graph);
    f.expect(f.a, 0);
    f.expect(f.b, 12);
    f.expect(f.c, 0);
    f.expect(f.output, 13);
}

void graph_embedding_is_transactional(Fixture& f) {
    Graph source;
    {
        auto record = source.record_guard(&f.stream);
        f.emit(f.a, 99);
        f.stream.copy(f.a.view(), f.b.view());
        record.finish();
    }
    Graph target;
    {
        auto record = target.record_guard(&f.stream);
        auto predicate = f.stream.predicate_scope(f.one.view());
        f.emit(f.c, 1);
        throws([&] { source.launch(&f.stream); },
               "copy-containing graph was accepted into a predicate body");
        f.emit(f.output, 14);
        predicate.finish();
        record.finish();
    }
    check(target.size() == 2, "failed graph embedding left a partially appended graph");
    f.execute(target);
    f.expect(f.a, 0);
    f.expect(f.b, 0);
    f.expect(f.c, 1);
    f.expect(f.output, 14);

    Graph dependent;
    {
        auto record = dependent.record_guard(&f.stream);
        f.emit(f.a, 99);
        f.stream.dispatch(f.kernel, {1, 1, 1}, {1, 1, 1},
                          {Argument::buffer(0, f.a.view()),
                           Argument::buffer(1, f.b.view(), Access::Write),
                           Argument::scalar(2, Affine{1, 1, 0})});
        record.finish();
    }
    Graph independent;
    {
        auto record = independent.record_guard(&f.stream);
        auto parallel = f.stream.parallel();
        f.emit(f.c, 2);
        throws([&] { dependent.launch(&f.stream); },
               "dependent graph was accepted into a parallel group");
        f.emit(f.output, 15);
        parallel.finish();
        record.finish();
    }
    check(independent.size() == 2, "parallel embedding failure left partial work");
    f.execute(independent);
    f.expect(f.a, 0);
    f.expect(f.b, 0);
    f.expect(f.c, 2);
    f.expect(f.output, 15);
}

void transaction_host_boundary_is_rejected(Fixture& f) {
    auto transaction = f.stream.transaction();
    f.emit(f.output, 99);
    bool called = false;
    throws([&] { f.stream.host([&] { called = true; }); },
           "transaction allowed an irreversible host boundary");
    check(!called && f.stream.statistics().command_buffers == 0,
          "host boundary executed before transaction commit");
    transaction.cancel();
    f.emit(f.output, 16);
    f.stream.synchronize();
    f.expect(f.output, 16);
}

void internal_control_edits_are_rejected(Fixture& f) {
    Graph graph;
    {
        auto record = graph.record_guard(&f.stream);
        if_else(f.stream, {f.one.view()}, [&] { f.emit(f.output, 17); });
        record.finish();
    }
    check(graph.commands()[0].compute.control, "expected a runtime control command");
    auto update = graph.update();
    throws([&] { update.set_grid(0, {2, 1, 1}, {1, 1, 1}); },
           "individual runtime control-grid edit was accepted");
    throws([&] { update.set_buffer(0, 0, f.zero.view()); },
           "individual runtime control-buffer edit was accepted");
    auto replacement = f.device.allocate(4);
    replacement.data<uint32_t>()[0] = 1;
    update.rebind(f.one, replacement);
    update.commit();
    f.execute(graph);
    f.expect(f.output, 17);
}
} // namespace

int main(int argc, char** argv) {
    try {
        Device device;
        if (argc > 1 && std::string{argv[1]} == "--metal") {
            try {
                device = Device::system_default();
            } catch (const std::exception& e) {
                std::cout << "SKIP: " << e.what() << '\n';
                return 77;
            }
        } else
            device = Device::reference();
        const auto cases = {stale_scopes_after_batch_cancel,
                            stale_scopes_after_capture_cancel,
                            out_of_order_predicates,
                            out_of_order_transactions,
                            live_transaction_blocks_parent_finish,
                            cancel_restores_parallel_state,
                            graph_embedding_is_transactional,
                            transaction_host_boundary_is_rejected,
                            internal_control_edits_are_rejected};
        size_t passed = 0;
        for (auto policy :
             {DispatchPolicy::PerKernel, DispatchPolicy::Direct, DispatchPolicy::Indirect}) {
            if (policy == DispatchPolicy::Indirect && !device.capabilities().indirect_compute)
                continue;
            for (auto test : cases) {
                Fixture fixture{device, policy};
                test(fixture);
                ++passed;
            }
        }
        std::cout << "PASS: " << passed << " recording scope safety cases\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "FAIL: " << e.what() << '\n';
        return 1;
    }
}
