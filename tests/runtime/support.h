#pragma once
// Shared fixtures for runtime contract tests; the runner lives in test_runtime.cpp.
#include <metalrt/runtime.h>
#include <metalrt/detail/backend.h>
#include <metalrt/detail/cache_key.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <future>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

using namespace metalrt;
namespace runtime_tests {
struct TestSkipped : std::runtime_error {
    using std::runtime_error::runtime_error;
};
inline void check(bool condition, const std::string& message) {
    if (!condition)
        throw std::runtime_error(message);
}
inline void near(float actual, float expected, const std::string& message) {
    if (!std::isfinite(actual) || !std::isfinite(expected) ||
        std::abs(actual - expected) > 1e-5f * std::max(1.0f, std::abs(expected)))
        throw std::runtime_error(message + ": got " + std::to_string(actual) + ", expected " +
                                 std::to_string(expected));
}
template <class F> void throws(F&& fn, const std::string& message) {
    bool raised = false;
    try {
        fn();
    } catch (const std::exception&) {
        raised = true;
    }
    check(raised, message);
}
struct AffineParams {
    uint32_t n;
    float scale, bias;
};
static_assert(sizeof(AffineParams) == 12, "MSL constant ABI");
#ifndef METALRT_TEST_SHADER_DIR
#define METALRT_TEST_SHADER_DIR "kernels"
#endif
inline const std::string shader_file =
    std::string{METALRT_TEST_SHADER_DIR} + "/runtime_tests.metal";
inline CpuKernel affine_reference() {
    return [](const ComputeCommand& c) {
        auto p = c.scalar<AffineParams>(2);
        auto in = c.buffer<float>(0);
        auto out = c.buffer<float>(1);
        for (uint32_t i = 0; i < p.n; ++i)
            out[i] = p.scale * in[i] + p.bias;
    };
}
inline CpuKernel square_reference() {
    return [](const ComputeCommand& c) {
        auto n = c.scalar<uint32_t>(2);
        auto in = c.buffer<float>(0);
        auto out = c.buffer<float>(1);
        for (uint32_t i = 0; i < n; ++i)
            out[i] = in[i] * in[i];
    };
}
struct Kernels {
    Kernel affine, square, threadgroup_exchange;
    explicit Kernels(Device d)
        : affine(d.kernel_file(shader_file, "runtime_affine", affine_reference())),
          square(d.kernel_file(shader_file, "runtime_square", square_reference())),
          threadgroup_exchange(d.kernel_file(
              shader_file, "runtime_threadgroup_exchange", [](const ComputeCommand& c) {
                  uint32_t n = c.scalar<uint32_t>(2);
                  auto input = c.buffer<float>(0);
                  auto output = c.buffer<float>(1);
                  for (uint32_t i = 0; i < n; ++i)
                      output[i] = (i ^ 1u) < n ? input[i ^ 1u] : 0.0f;
              })) {}
};
inline void affine(Stream& s, const Kernel& k, BufferView input, BufferView output, uint32_t n,
                   float scale, float bias) {
    s.dispatch(k, {n, 1, 1}, {64, 1, 1},
               {Argument::buffer(0, std::move(input)),
                Argument::buffer(1, std::move(output), Access::Write),
                Argument::scalar(2, AffineParams{n, scale, bias})});
}
inline void square(Stream& s, const Kernel& k, BufferView input, BufferView output, uint32_t n) {
    s.dispatch(k, {n, 1, 1}, {64, 1, 1},
               {Argument::buffer(0, std::move(input)),
                Argument::buffer(1, std::move(output), Access::Write), Argument::scalar(2, n)});
}
inline const char* policy_name(DispatchPolicy policy) {
    switch (policy) {
    case DispatchPolicy::PerKernel:
        return "per-kernel";
    case DispatchPolicy::Direct:
        return "direct";
    case DispatchPolicy::Indirect:
        return "indirect";
    }
    return "invalid";
}

void policy_sequence(Device d, const Kernels& k, DispatchPolicy policy);
void pointers_and_validation(Device d, const Kernels& k);
void cancellation_and_nested_capture(Device d, const Kernels& k);
void temporary_lifetimes(Device d, const Kernels& k, DispatchPolicy policy);
void cpu_boundaries(Device d, const Kernels& k);
void private_memory_and_byte_transfers(Device d, const Kernels& k, DispatchPolicy policy);
void dynamic_threadgroup_memory(Device d, const Kernels& k, DispatchPolicy policy);
void capture_thread_rules(Device d, const Kernels& k);
void default_stream_capture(Device d, const Kernels& k);
void asynchronous_errors(Device d, const Kernels& k);
void deterministic_inflight_versioning(Device d);
void empty_recapture_releases_plan(Device d, const Kernels& k);
void nullable_buffers(Device d, DispatchPolicy policy);
void retained_stream_lifetime(Device d, const Kernels& kernels, DispatchPolicy policy);
void kernel_cache_contract(Device d, bool require_archive_hit = true);
} // namespace runtime_tests
