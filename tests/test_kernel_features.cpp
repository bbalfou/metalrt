#include <metalrt/runtime.h>
#include <metalrt/detail/cache_key.h>
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>

using namespace metalrt;
namespace {
void check(bool value, const char* message) {
    if (!value)
        throw std::runtime_error(message);
}
template <class F> void rejects(F f) {
    bool rejected = false;
    try {
        f();
    } catch (const std::exception&) {
        rejected = true;
    }
    check(rejected, "invalid kernel options accepted");
}
KernelCompileOptions options(float scale) {
    return {false,
            {FunctionConstant::scalar(0, scale), FunctionConstant::scalar(1, uint32_t{3}),
             FunctionConstant::scalar(2, true), FunctionConstant::scalar(3, int32_t{-1})}};
}
CpuKernel callback(float scale) {
    return [scale](const ComputeCommand& c) {
        for (size_t i = 0; i < c.grid.x; ++i)
            c.buffer<float>(0)[i] = float(i) * scale + 2;
    };
}
void execute(Device device, Kernel kernel, float scale, DispatchPolicy policy) {
    Stream stream{device, policy};
    auto output = device.allocate(64 * sizeof(float));
    Graph graph;
    {
        auto capture = graph.record_guard(&stream);
        stream.dispatch(kernel, {64, 1, 1}, {32, 1, 1},
                        {Argument::buffer(0, output.view(), Access::Write)});
        capture.finish();
    }
    graph.prepare(&stream);
    device.clear_kernel_cache(); // The graph retains the actual specialized pipeline.
    graph.launch(&stream).wait();
    for (size_t i = 0; i < 64; ++i)
        check(output.data<float>()[i] == float(i) * scale + 2, "specialized result mismatch");
}
} // namespace
int main(int argc, char** argv) {
    try {
        bool offline = argc > 1 && std::string(argv[1]) == "--metallib-only";
        bool native = offline || (argc > 1 && std::string(argv[1]) == "--metal");
        Device device;
        if (native) {
            try {
                device = Device::system_default();
            } catch (const std::exception& e) {
                std::cout << "SKIP: " << e.what() << '\n';
                return 77;
            }
        } else
            device = Device::reference();
#ifdef METALRT_TEST_METALLIB
        if (offline) {
            check(device.kernel_cache_statistics().source_compilations == 0,
                  "fresh process already compiled source");
            auto kernel =
                device.kernel_metallib(METALRT_TEST_METALLIB, "specialized_value", options(2));
            check(device.kernel_cache_statistics().source_compilations == 0,
                  "offline loading compiled source");
            execute(device, kernel, 2, DispatchPolicy::Direct);
            std::cout << "PASS: fresh process loads specialized metallib without MSL compilation\n";
            return 0;
        }
#endif
        std::ifstream file(std::string(METALRT_TEST_SHADER_DIR) + "/function_constants.metal");
        std::string source{std::istreambuf_iterator<char>(file), {}};
        check(!source.empty(), "missing shader fixture");
        auto a = options(2), b = options(5);
        auto reversed = a;
        std::reverse(reversed.constants.begin(), reversed.constants.end());
        check(detail::compile_options_key(detail::validated_compile_options(a)) ==
                  detail::compile_options_key(detail::validated_compile_options(reversed)),
              "order-dependent key");
        check(detail::compile_options_key(a) != detail::compile_options_key(b),
              "constant key collision");
        auto bad = a;
        bad.constants.push_back(bad.constants.front());
        rejects([&] { device.kernel(source, "specialized_value", bad, callback(2)); });
        bad = a;
        bad.constants[0].value.clear();
        rejects([&] { device.kernel(source, "specialized_value", bad, callback(2)); });
        bad = a;
        bad.constants[0].index = 65536;
        rejects([&] { device.kernel(source, "specialized_value", bad, callback(2)); });
        bad = a;
        bad.constants[0].type = static_cast<FunctionConstant::Type>(999);
        rejects([&] { device.kernel(source, "specialized_value", bad, callback(2)); });
        device.clear_kernel_cache();
        auto before = device.kernel_cache_statistics();
        auto first = device.kernel(source, "specialized_value", a, callback(2));
        auto second = device.kernel(source, "specialized_value", b, callback(5));
        auto same =
            device.kernel_file(std::string(METALRT_TEST_SHADER_DIR) + "/function_constants.metal",
                               "specialized_value", reversed, callback(2));
        if (native) {
            check(first == same && !(first == second), "specialized pipeline identity incorrect");
            check(device.kernel_cache_statistics().source_compilations ==
                      before.source_compilations + 1,
                  "constants recompiled the common library");
            bad = a;
            bad.constants[0] = FunctionConstant::scalar(0, uint32_t{2});
            rejects([&] { device.kernel(source, "specialized_value", bad); });
            bad = a;
            bad.constants[0].index = 99;
            rejects([&] { device.kernel(source, "specialized_value", bad); });
            rejects([&] { device.kernel(source, "missing_entry", a); });
            rejects([&] { device.kernel(source, "specialized_value", KernelCompileOptions{}); });
        }
        if (native) {
            const auto root =
                std::filesystem::temp_directory_path() /
                ("metalrt-specializations-" +
                 std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
            struct Cleanup {
                std::filesystem::path path;
                ~Cleanup() {
                    std::error_code ignored;
                    std::filesystem::remove_all(path, ignored);
                }
            } cleanup{root};
            auto cached_a = device.kernel_cached(source, "specialized_value", root.string(), a);
            auto cached_b = device.kernel_cached(source, "specialized_value", root.string(), b);
            size_t files = 0;
            for (const auto& entry : std::filesystem::directory_iterator(root))
                if (entry.path().extension() == ".metallib")
                    ++files;
            check(files == 2, "persistent specializations shared an archive key");
            device.clear_kernel_cache();
            auto reloaded_a = device.kernel_cached(source, "specialized_value", root.string(), a);
            auto reloaded_b = device.kernel_cached(source, "specialized_value", root.string(), b);
            execute(device, reloaded_a, 2, DispatchPolicy::Direct);
            execute(device, reloaded_b, 5, DispatchPolicy::Direct);
        }
        for (auto policy :
             {DispatchPolicy::Direct, DispatchPolicy::PerKernel, DispatchPolicy::Indirect}) {
            if (policy == DispatchPolicy::Indirect && !device.capabilities().indirect_compute)
                continue;
            execute(device, first, 2, policy);
            execute(device, second, 5, policy);
        }
#ifdef METALRT_TEST_METALLIB
        if (native) {
            device.clear_kernel_cache();
            before = device.kernel_cache_statistics();
            auto precompiled =
                device.kernel_metallib(METALRT_TEST_METALLIB, "specialized_value", a);
            auto other = device.kernel_metallib(METALRT_TEST_METALLIB, "specialized_value", b);
            check(device.kernel_cache_statistics().source_compilations ==
                      before.source_compilations,
                  "metallib loading compiled MSL source");
            check(!(precompiled == other), "metallib specialization collision");
            for (auto policy :
                 {DispatchPolicy::Direct, DispatchPolicy::PerKernel, DispatchPolicy::Indirect}) {
                if (policy == DispatchPolicy::Indirect && !device.capabilities().indirect_compute)
                    continue;
                execute(device, precompiled, 2, policy);
                execute(device, other, 5, policy);
            }
            const auto temporary =
                std::filesystem::temp_directory_path() /
                ("metalrt-library-" +
                 std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
            struct RemoveFile {
                std::filesystem::path path;
                ~RemoveFile() {
                    std::error_code ignored;
                    std::filesystem::remove(path, ignored);
                }
            } remove_file{temporary};
            std::filesystem::copy_file(METALRT_TEST_METALLIB, temporary);
            auto retained_library =
                device.kernel_metallib(temporary.string(), "specialized_value", a);
            {
                std::ofstream replace(temporary, std::ios::binary | std::ios::trunc);
                replace << "corrupt";
            }
            rejects([&] { device.kernel_metallib(temporary.string(), "specialized_value", a); });
            execute(device, retained_library, 2, DispatchPolicy::Direct);
            rejects([&] { device.kernel_metallib(METALRT_TEST_METALLIB, "missing", a); });
            rejects([&] {
                device.kernel_metallib(std::string(METALRT_TEST_SHADER_DIR) +
                                           "/function_constants.metal",
                                       "specialized_value", a);
            });
        }
#endif
        rejects(
            [&] { device.kernel_metallib("missing-library.metallib", "specialized_value", a); });
        std::cout << "PASS: function constants, shared library reuse, retained variants and "
                     "metallib loading\n";
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
