#include "example_support.h"

using namespace metalrt_examples;
int main(int argc, char** argv) {
    try {
        const auto options = settings(argc, argv, "parallel_branches");
        auto device = select_device(options);
        Stream stream{device, options.policy};
        constexpr uint32_t count = 257;
        auto input = device.allocate(count * sizeof(float)),
             x = device.allocate(count * sizeof(float));
        // Two disjoint views of one allocation demonstrate explicit range checks.
        auto branches = device.allocate(2 * count * sizeof(float)),
             output = device.allocate(count * sizeof(float));
        auto y = branches.view(0, count * sizeof(float)),
             z = branches.view(count * sizeof(float), count * sizeof(float));
        for (uint32_t i = 0; i < count; ++i)
            input.data<float>()[i] = float(i) / 32;
        auto transform = affine_kernel(device), sum = sum_kernel(device);
        Graph graph;
        {
            auto recording = graph.record_guard(&stream);
            affine(stream, transform, input.view(), x.view(), count, 2, 1); // A
            {
                auto parallel = stream.parallel();
                affine(stream, transform, x.view(), y, count, 3, 0);  // B
                affine(stream, transform, x.view(), z, count, -1, 4); // C
                parallel.finish();
            }
            launch_linear(&stream, sum, count,
                          {// D
                           Argument::buffer(0, y), Argument::buffer(1, z),
                           Argument::buffer(2, output.view(), Access::Write),
                           Argument::scalar(3, count)});
            recording.finish();
        }
        graph.prepare(&stream);
        graph.launch(&stream).wait();
        stream.synchronize();
        for (uint32_t i = 0; i < count; ++i)
            near(output.data<float>()[i], 4 * input.data<float>()[i] + 6);
        const auto commands = graph.commands();
        if (!commands[1].compute.barrier_before || commands[2].compute.barrier_before ||
            !commands[3].compute.barrier_before)
            throw std::runtime_error("parallel entry/join barriers were not recorded");
        std::cout << "A -> (B, C) -> D verified for " << count << " values.\n"
                  << "B/C share read-only X and write disjoint Y/Z views; D joins both branches.\n"
                  << "Indirect policy permits B/C GPU overlap. Permission does not guarantee "
                     "physical overlap or speedup.\n"
                  << "Native direct/per-kernel dispatch stays ordered; reference scheduling "
                     "validates dependencies without measuring GPU overlap.\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
