#include "example_support.h"

using namespace metalrt_examples;
int main(int argc, char** argv) {
    try {
        const auto options = settings(argc, argv, "replay_updates");
        auto device = select_device(options);
        Stream stream{device, options.policy};
        constexpr uint32_t count = 37;
        auto first = device.allocate(count * sizeof(float)),
             second = device.allocate(count * sizeof(float));
        auto output = device.allocate(count * sizeof(float));
        for (uint32_t i = 0; i < count; ++i) {
            first.data<float>()[i] = float(i);
            second.data<float>()[i] = float(i) + 100;
            output.data<float>()[i] = -1;
        }
        auto kernel = affine_kernel(device);
        Graph graph;
        {
            // Recording and preparation do not execute the pipeline.
            auto recording = graph.record_guard(&stream);
            affine(stream, kernel, first.view(), output.view(), count, 2, 1);
            recording.finish();
        }
        graph.prepare(&stream);
        near(output.data<float>()[0], -1);
        graph.launch(&stream).wait();
        for (uint32_t i = 0; i < count; ++i)
            near(output.data<float>()[i], 2 * first.data<float>()[i] + 1);

        // The public transaction changes both arguments atomically. No capture
        // code or kernel compilation runs again. Previous launches keep their
        // own command snapshot and resources until completion.
        auto update = graph.update();
        update.set_buffer(0, 0, second.view()).set_scalar(0, 2, AffineArgs{count, 0.5f, -3});
        update.commit();
        graph.launch(&stream).wait();
        for (uint32_t i = 0; i < count; ++i)
            near(output.data<float>()[i], 0.5f * second.data<float>()[i] - 3);

        // A clone starts with the same bindings, then accepts independent edits.
        auto variant = graph.clone();
        auto variant_output = device.allocate(count * sizeof(float));
        auto variant_update = variant.update();
        variant_update.set_buffer(0, 1, variant_output.view())
            .set_scalar(0, 2, AffineArgs{count, 3, 4});
        variant_update.commit();
        variant.launch(&stream).wait();
        graph.launch(&stream).wait();
        stream.synchronize();
        for (uint32_t i = 0; i < count; ++i) {
            near(variant_output.data<float>()[i], 3 * second.data<float>()[i] + 4);
            near(output.data<float>()[i], 0.5f * second.data<float>()[i] - 3);
        }
        const auto stats = stream.statistics();
        std::cout
            << "Prepared without execution; scalar/buffer update and independent clone verified.\n"
            << "Graph revision: " << graph.revision()
            << "\nICB builds/updates/reuses: " << stats.icb_builds << '/' << stats.icb_updates
            << '/' << stats.icb_reuses << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
