// Completion contracts across supported backends and dispatch policies.
#include "support.h"
namespace runtime_tests {
void asynchronous_errors(Device d, const Kernels& k) {
    if (d.capabilities().backend != BackendKind::Reference)
        return;
    Stream stream(d);
    auto input = d.allocate(4), output = d.allocate(4);
    *input.data<float>() = 3;
    auto fail = d.kernel_file(shader_file, "runtime_affine", [](const ComputeCommand&) {
        throw std::runtime_error("intentional asynchronous kernel failure");
    });
    affine(stream, fail, input.view(), output.view(), 1, 1, 0);
    throws([&] { stream.last_ticket().wait(); }, "ticket must propagate an asynchronous error");
    {
        auto batch = stream.batch();
        affine(stream, k.affine, input.view(), output.view(), 1, 2, 0);
        // finish discovers the previous failed ticket. The destructor must defer
        // the exception rather than terminate the process or throw from noexcept.
    }
    throws([&] { stream.check_errors(); }, "RAII submission errors must be deferred to the stream");
    throws([&] { stream.synchronize(); }, "synchronize observes and retires failed work");
    affine(stream, k.affine, input.view(), output.view(), 1, 2, 0);
    stream.synchronize();
    near(*output.data<float>(), 6, "stream remains usable after error observation");
    Graph graph;
    {
        auto g = graph.capture_guard(&stream);
        affine(stream, k.affine, input.view(), output.view(), 1, 2, 1);
    }
    stream.synchronize();
    const auto previous = graph.commands();
    affine(stream, fail, input.view(), output.view(), 1, 1, 0);
    throws([&] { stream.last_ticket().wait(); }, "second failing ticket");
    {
        auto g = graph.capture_guard(&stream);
        affine(stream, k.affine, input.view(), output.view(), 1, 9, 9);
        throws([&] { g.finish(); }, "failed submit must propagate from explicit finish");
    }
    check(detail::same_commands(graph.commands(), previous),
          "failed capture submit must preserve the previous successful snapshot");
    throws([&] { stream.synchronize(); },
           "failed capture cleanup retires the preceding failed ticket");
}

void deterministic_inflight_versioning(Device d) {
    if (d.capabilities().backend != BackendKind::Reference)
        return;
    Stream stream(d, DispatchPolicy::Indirect);
    std::promise<void> release_promise, started_promise;
    auto release = release_promise.get_future().share();
    auto started = started_promise.get_future();
    std::atomic<bool> first{true};
    struct ReleaseOnExit {
        std::promise<void>& promise;
        Stream& stream;
        bool released = false;
        void finish() {
            if (!released) {
                released = true;
                promise.set_value();
            }
        }
        ~ReleaseOnExit() {
            try {
                finish();
            } catch (...) {
            }
            try {
                stream.synchronize();
            } catch (...) {
            }
        }
    } release_on_exit{release_promise,
                      stream}; // Destroyed before stream; failures cannot deadlock its destructor.
    auto gated = d.kernel_file(shader_file, "runtime_affine", [&](const ComputeCommand& c) {
        if (first.exchange(false))
            started_promise.set_value();
        release.wait();
        affine_reference()(c);
    });
    auto input = d.allocate(4), output1 = d.allocate(4), output2 = d.allocate(4);
    *input.data<float>() = 3;
    Graph graph;
    {
        auto g = graph.capture_guard(&stream);
        affine(stream, gated, input.view(), output1.view(), 1, 1, 0);
    }
    check(started.wait_for(std::chrono::seconds(5)) == std::future_status::ready,
          "reference worker failed to begin the gated command");
    std::weak_ptr<detail::BufferAllocation> retained = input.implementation();
    {
        auto g = graph.capture_guard(&stream);
        affine(stream, gated, input.view(), output2.view(), 1, 2, 1);
    }
    check(stream.statistics().icb_builds == 2,
          "changed in-flight graph must allocate a distinct executable version");
    graph.reset();
    input = {};
    check(!retained.expired(), "queued submissions must keep released input alive");
    release_on_exit.finish();
    stream.synchronize();
    near(*output1.data<float>(), 3, "old submission keeps old scalar and binding");
    near(*output2.data<float>(), 7, "new submission receives new scalar and binding");
    check(retained.expired(),
          "retired reference submissions should release their buffer ownership");
}

void retained_stream_lifetime(Device d, const Kernels& kernels, DispatchPolicy policy) {
    auto owner = std::make_unique<Stream>(d, policy, StreamOrdering::Independent);
    auto retained = owner->retain();
    check(retained->device() == d && retained->policy() == policy &&
              retained->ordering() == StreamOrdering::Independent,
          "retained alias must preserve its queue/device/ordering");
    auto input = d.allocate(sizeof(float)), output = d.allocate(sizeof(float));
    *input.data<float>() = 2;
    Graph graph;
    {
        auto capture = graph.record_guard(owner.get());
        check(retained->is_capturing(), "retained aliases must see the original graph recording");
        auto temporary = retained->retain();
        affine(*temporary, kernels.affine, input.view(), output.view(), 1, 3, 1);
        const auto waits = owner->statistics().stream_wait_calls;
        temporary.reset();
        check(owner->statistics().stream_wait_calls == waits,
              "releasing a nonlast retained alias must not synchronize");
    }
    check(graph.size() == 1 && !retained->is_recording(),
          "retained dispatch must join and leave the original capture");
    graph.launch(retained.get());
    owner.reset(); // Existing owner destruction still waits for its issued work.
    near(*output.data<float>(), 7, "original stream destruction keeps completion semantics");
    affine(*retained, kernels.affine, output.view(), input.view(), 1, 2, 1);
    retained.reset(); // Last alias owns retirement of work submitted afterwards.
    near(*input.data<float>(), 15, "last retained alias must retire subsequent work");
    // The original owner can also die while its queue waits for a cancellable
    // producer. A deferred destructor must not erase later alias submissions.
    Stream producer(d, policy, StreamOrdering::Default);
    auto dependent = std::make_unique<Stream>(d, policy, StreamOrdering::Blocking);
    auto continuation = dependent->retain();
    auto final = d.allocate(sizeof(float));
    {
        auto batch = producer.batch();
        producer.fill(input.view(), 0);
        affine(*dependent, kernels.affine, input.view(), output.view(), 1, 1, 1);
        dependent.reset();
        affine(*continuation, kernels.affine, output.view(), final.view(), 1, 2, 1);
    }
    continuation->synchronize();
    near(*final.data<float>(), 3,
         "retained alias preserves deferred producer ordering after original destruction");
}

} // namespace runtime_tests
