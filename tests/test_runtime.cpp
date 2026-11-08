// Test runner shared by reference, Metal, and binary archive reuse gates.
#include "runtime/support.h"
using namespace runtime_tests;
int main(int argc, char** argv) {
    const bool cache_only = argc > 1 && std::string(argv[1]) == "--metal-cache";
    const bool native = cache_only || (argc > 1 && std::string(argv[1]) == "--metal");
    Device device;
    try {
        device = native ? Device::system_default() : Device::reference();
    } catch (const std::exception& e) {
        std::cout << "SKIP: " << e.what() << '\n';
        return 77;
    }
    Device::set_current(device);
    if (cache_only) {
        try {
            kernel_cache_contract(device, true);
            std::cout << "PASS native binary archive reuse and recovery\n";
            return 0;
        } catch (const std::exception& e) {
            std::cerr << "FAIL native binary archive: " << e.what() << '\n';
            return 1;
        }
    }
    int passed = 0, skipped = 0, failed = 0;
    auto run = [&](const std::string& name, auto fn) {
        try {
            fn();
            ++passed;
            std::cout << "PASS " << name << '\n';
        } catch (const TestSkipped& e) {
            ++skipped;
            std::cout << "SKIP " << name << ": " << e.what() << '\n';
        } catch (const std::exception& e) {
            ++failed;
            std::cerr << "FAIL " << name << ": " << e.what() << '\n';
        }
    };
    try {
        Kernels kernels(device);
        std::cout << "Backend: " << device.capabilities().name << '\n';
        for (auto policy :
             {DispatchPolicy::PerKernel, DispatchPolicy::Direct, DispatchPolicy::Indirect}) {
            if (policy == DispatchPolicy::Indirect && !device.capabilities().indirect_compute) {
                std::cout << "SKIP indirect policy: compute ICB support unavailable\n";
                continue;
            }
            const std::string label = policy_name(policy);
            run(label + " dependent sequence, replay, updates",
                [&] { policy_sequence(device, kernels, policy); });
            run(label + " retained temporary lifetime",
                [&] { temporary_lifetimes(device, kernels, policy); });
            run(label + " private memory and byte transfers",
                [&] { private_memory_and_byte_transfers(device, kernels, policy); });
            run(label + " dynamic threadgroup memory",
                [&] { dynamic_threadgroup_memory(device, kernels, policy); });
            run(label + " nullable bindings and graph updates",
                [&] { nullable_buffers(device, policy); });
            run(label + " retained queue lifetime",
                [&] { retained_stream_lifetime(device, kernels, policy); });
        }
        run("pointer registry and validation", [&] { pointers_and_validation(device, kernels); });
        run("cancellation, nested scopes, synchronization",
            [&] { cancellation_and_nested_capture(device, kernels); });
        run("explicit CPU boundaries", [&] { cpu_boundaries(device, kernels); });
        run("capture thread ownership", [&] { capture_thread_rules(device, kernels); });
        run("default stream capture", [&] { default_stream_capture(device, kernels); });
        if (device.capabilities().backend == BackendKind::Reference) {
            run("asynchronous errors and destructor error deferral",
                [&] { asynchronous_errors(device, kernels); });
            run("deterministic in-flight executable versioning",
                [&] { deterministic_inflight_versioning(device); });
        }
        run("empty capture releases executable resources",
            [&] { empty_recapture_releases_plan(device, kernels); });
        const char* shader_validation = std::getenv("MTL_SHADER_VALIDATION");
        const bool instrumented = shader_validation && std::string(shader_validation) == "1";
        run("kernel cache identity, persistence, and recovery",
            [&] { kernel_cache_contract(device, !instrumented); });
    } catch (const std::exception& e) {
        ++failed;
        std::cerr << "FAIL setup: " << e.what() << '\n';
    }
    std::cout << passed << " passed, " << skipped << " skipped, " << failed << " failed\n";
    return failed ? 1 : 0;
}
