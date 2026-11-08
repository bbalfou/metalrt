// Cache contracts across supported backends and dispatch policies.
#include "support.h"
namespace runtime_tests {
void kernel_cache_contract(Device d, bool require_archive_hit) {
    for (unsigned overload = 0; overload < 3; ++overload) {
        // Use a fresh non-global backend so its lifetime is observable. The
        // public handle, including copies retained by commands, must own the
        // device identity without making the backend's cache cyclic.
        Device isolated{detail::make_reference_device()};
        std::weak_ptr<detail::DeviceBackend> owner = isolated.implementation();
        const auto callback = [](const ComputeCommand&) {};
        Kernel retained;
        if (overload == 0)
            retained = isolated.kernel("unused", "lifetime", callback);
        else if (overload == 1)
            retained = isolated.kernel_cached("unused", "lifetime", {}, callback);
        else
            retained =
                isolated.kernel_cached("unused", "lifetime", {}, KernelCompileOptions{}, callback);
        isolated = {};
        check(!owner.expired(), "kernel handle must retain its device identity");
        auto copy = retained;
        retained = {};
        check(!owner.expired(), "copied command kernel must retain its device identity");
        copy = {};
        check(owner.expired(), "kernel ownership must not create a device/cache cycle");
    }
    check(detail::kernel_cache_digest("") ==
              "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
          "empty SHA-256 cache key");
    check(detail::kernel_cache_digest("abc") ==
              "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
          "short SHA-256 cache key");
    check(detail::kernel_cache_digest("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq") ==
              "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1",
          "two-block SHA-256 padding");
    check(detail::kernel_cache_digest(std::string(1000000, 'a')) ==
              "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0",
          "multi-block SHA-256 cache key");
    const bool native = d.capabilities().backend == BackendKind::Metal;
    check(d.capabilities().runtime_compilation == native,
          "runtime compilation capability is backend-specific");
    check(d.capabilities().binary_archives == native,
          "binary archive capability is backend-specific");
    const auto root =
        std::filesystem::current_path() /
        (native ? "metalrt-kernel-cache-metal-test" : "metalrt-kernel-cache-reference-test");
    std::filesystem::remove_all(root);
    const std::string source = "#include <metal_stdlib>\nusing namespace metal;\nkernel void "
                               "cached_value(device float* out [[buffer(0)]]) { out[0]=17.0f; }\n";
    auto callback = [](const ComputeCommand& command) { command.buffer<float>(0)[0] = 17; };
    auto first_request = std::async(std::launch::async, [&] {
        return d.kernel_cached(source, "cached_value", root.string(), callback);
    });
    auto second_request = std::async(std::launch::async, [&] {
        return d.kernel_cached(source, "cached_value", root.string(), callback);
    });
    auto first = first_request.get(), second = second_request.get();
    check(first.max_threads_per_threadgroup() > 0,
          "compiled kernel must expose its threadgroup limit");
    check((first.thread_execution_width() > 0) == native,
          "SIMD execution width must describe native hardware only");
    if (!native) {
        check(!(first == second), "reference callbacks must not be deduplicated by MSL source");
        check(!std::filesystem::exists(root),
              "reference backend must not fabricate compiled artifacts");
        check(d.kernel_cache_statistics().archive_hits == 0,
              "reference backend must not claim native cache hits");
        d.clear_kernel_cache();
        return;
    }
    check(first == second, "same device/source/entry must reuse retained pipeline");
    check(first.max_threads_per_threadgroup() <= d.capabilities().max_threads_per_threadgroup,
          "pipeline thread limit cannot exceed the device limit");
    check(first.static_threadgroup_memory_bytes() <= d.capabilities().max_threadgroup_memory_bytes,
          "compiled static threadgroup storage cannot exceed device memory");
    auto stats = d.kernel_cache_statistics();
    check(stats.memory_hits > 0 && stats.archive_writes > 0,
          "native cache did not record memory hit and binary write");
    std::filesystem::path archive;
    for (const auto& file : std::filesystem::directory_iterator(root))
        if (file.path().extension() == ".metallib")
            archive = file.path();
    check(!archive.empty() && std::filesystem::file_size(archive) > 0,
          "native archive artifact is missing");
    auto fast = d.kernel_cached(source, "cached_value", root.string(), KernelCompileOptions{true},
                                callback);
    check(!(fast == first), "compiler math options must distinguish in-memory pipeline keys");
    size_t archive_count = 0;
    for (const auto& file : std::filesystem::directory_iterator(root))
        archive_count += file.path().extension() == ".metallib";
    check(archive_count == 2, "compiler math options must distinguish persistent archive keys");
    auto output = d.allocate(sizeof(float));
    Stream stream(d);
    Graph graph;
    {
        auto recording = graph.record_guard(&stream);
        stream.dispatch(first, {1, 1, 1}, {1, 1, 1},
                        {Argument::buffer(0, output.view(), Access::Write)});
    }
    d.clear_kernel_cache();
    graph.launch(&stream).wait();
    near(*output.data<float>(), 17, "recorded pipeline survives cache eviction");
    stats = d.kernel_cache_statistics();
    auto loaded = d.kernel_cached(source, "cached_value", root.string(), callback);
    const auto reloaded = d.kernel_cache_statistics();
    // GPU validation may instrument the pipeline after archive harvesting.
    // Exercise recovery here; the separate uninstrumented cache gate requires
    // a real binary hit, so recompilation cannot masquerade as cache reuse.
    if (require_archive_hit)
        check(reloaded.archive_hits > stats.archive_hits,
              "cold in-memory cache did not load native archive");
    else
        check(reloaded.archive_hits > stats.archive_hits ||
                  reloaded.archive_misses > stats.archive_misses,
              "cold in-memory cache did not attempt native archive recovery");
    stream.dispatch(loaded, {1, 1, 1}, {1, 1, 1},
                    {Argument::buffer(0, output.view(), Access::Write)});
    stream.synchronize();
    near(*output.data<float>(), 17, "archive-loaded pipeline result");
    d.clear_kernel_cache();
    std::ofstream(archive, std::ios::binary | std::ios::trunc) << "invalid binary archive";
    stats = d.kernel_cache_statistics();
    auto repaired = d.kernel_cached(source, "cached_value", root.string(), callback);
    check(d.kernel_cache_statistics().archive_errors > stats.archive_errors,
          "corrupt archive must be diagnosed");
    check(d.kernel_cache_statistics().archive_writes > stats.archive_writes,
          "corrupt archive must be replaced after recompilation");
    stream.dispatch(repaired, {1, 1, 1}, {1, 1, 1},
                    {Argument::buffer(0, output.view(), Access::Write)});
    stream.synchronize();
    near(*output.data<float>(), 17, "repaired archive pipeline result");
    std::ofstream(root / "not-a-directory") << "occupied";
    d.clear_kernel_cache();
    stats = d.kernel_cache_statistics();
    auto uncached =
        d.kernel_cached(source, "cached_value", (root / "not-a-directory").string(), callback);
    check(d.kernel_cache_statistics().archive_errors > stats.archive_errors,
          "invalid cache destination must be observable");
    stream.dispatch(uncached, {1, 1, 1}, {1, 1, 1},
                    {Argument::buffer(0, output.view(), Access::Write)});
    stream.synchronize();
    near(*output.data<float>(), 17, "cache I/O failure cannot disable execution");
    const std::string changed = "#include <metal_stdlib>\nusing namespace metal;\nkernel void "
                                "cached_value(device float* out [[buffer(0)]]) { out[0]=19.0f; }\n";
    auto updated = d.kernel_cached(changed, "cached_value", root.string());
    stream.dispatch(updated, {1, 1, 1}, {1, 1, 1},
                    {Argument::buffer(0, output.view(), Access::Write)});
    stream.synchronize();
    near(*output.data<float>(), 19, "changed source must invalidate old compiled binary");
    std::filesystem::remove_all(root);
}
} // namespace runtime_tests
