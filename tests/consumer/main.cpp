// This consumer includes and links only the installed standalone runtime.
#include <metalrt/runtime.h>
#include <iostream>
#include <string>

int main(int argc, char** argv) {
    try {
        bool native = argc > 1 && std::string(argv[1]) == "--metal";
        auto device = native ? metalrt::Device::system_default() : metalrt::Device::reference();
        metalrt::Device::set_current(device);
        auto buffer = device.allocate(sizeof(float));
        buffer.data<float>()[0] = 7.0f;
        auto kernel = device.kernel_file(
            CONSUMER_SHADER_FILE, "consumer_twice",
            [](const metalrt::ComputeCommand& command) { command.buffer<float>(0)[0] *= 2; });
        metalrt::Stream stream{device};
        metalrt::launch_linear(
            &stream, kernel, 1,
            {metalrt::Argument::buffer(0, buffer.view(), metalrt::Access::ReadWrite)}, 1);
        stream.synchronize();
        if (buffer.data<float>()[0] != 14.0f)
            return 1;
        std::cout << "Installed standalone metalrt consumer passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
