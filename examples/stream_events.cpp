#include "example_support.h"

using namespace metalrt_examples;
int main(int argc, char** argv) {
    try {
        const auto options = settings(argc, argv, "stream_events");
        auto device = select_device(options);
        Stream producer{device, options.policy}, consumer{device, options.policy};
        TimelineEvent ready{device};
        constexpr uint32_t count = 73;
        auto input = device.allocate(count * sizeof(float)),
             shared = device.allocate(count * sizeof(float));
        auto output = device.allocate(count * sizeof(float));
        for (uint32_t i = 0; i < count; ++i)
            input.data<float>()[i] = float(i) / 16;
        auto kernel = affine_kernel(device);
        Graph produce, consume;
        {
            auto recording = produce.record_guard(&producer);
            affine(producer, kernel, input.view(), shared.view(), count, 2, 1);
            recording.finish();
        }
        {
            auto recording = consume.record_guard(&consumer);
            affine(consumer, kernel, shared.view(), output.view(), count, 3, -2);
            recording.finish();
        }
        produce.prepare(&producer);
        consume.prepare(&consumer);

        produce.launch(&producer);
        const auto first = producer.signal(ready);
        // This attaches a fresh GPU dependency to this submission. The event
        // value is never frozen into the reusable graph/ICB itself.
        consume.launch(&consumer, {first});
        if (producer.statistics().stream_wait_calls || consumer.statistics().stream_wait_calls)
            throw std::runtime_error("unexpected CPU synchronization during event submission");
        consumer.synchronize(); // Only the final CPU read needs this boundary.
        for (uint32_t i = 0; i < count; ++i)
            near(output.data<float>()[i], 6 * input.data<float>()[i] + 1);

        // A second replay gets a new monotonically increasing point. Stream's
        // standalone wait() also orders subsequent commands without a CPU wait.
        produce.launch(&producer);
        const auto second = producer.signal(ready);
        consumer.wait(second);
        consume.launch(&consumer);
        consumer.synchronize();
        producer.synchronize();
        for (uint32_t i = 0; i < count; ++i)
            near(output.data<float>()[i], 6 * input.data<float>()[i] + 1);
        if (second.value() <= first.value() || ready.completed_value() < second.value())
            throw std::runtime_error("timeline value did not advance");
        std::cout << "Two cross-stream submissions verified, with event points " << first.value()
                  << " and " << second.value() << ".\n"
                  << "Producer event signals: " << producer.statistics().event_signals
                  << "; consumer GPU waits: " << consumer.statistics().event_waits << '\n'
                  << "One producer stream owns this timeline; use a separate event for another "
                     "producer.\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
