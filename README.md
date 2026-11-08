# metalrt

A C++17 compute runtime for recording and replaying work on Apple Metal.

`metalrt` grew out of porting **tiny-cuda-nn** and replacing its use of CUDA
Graph capture. Metal has no direct equivalent to CUDA stream capture: this
runtime records its own commands, retains their resources, and prepares them
for repeated submission. It is standalone and has no dependency.

## Compatibility target

- **Primary target:** tiny-cuda-nn's guard-based capture/update/execute wrapper
  behavior. Preserve `capture_guard()` and its scope-based calling pattern.
- **Underlying model:** the subset of CUDA stream capture needed by the Metal port.
- **Additional facilities:** explicit replay, transactions, parallel groups,
  GPU control flow, rings, and parameters.

`capture_guard()` records work and submits it on successful scope completion.
`record_guard()` records without executing; `prepare()` and `launch()` expose
explicit preparation and replay as additional facilities.

Compatibility is behavioral and partial: this is not a complete CUDA API or ABI
replacement. Callers submit ported kernels through `metalrt::Stream`; arbitrary
native Metal calls are not intercepted.

See [capture compatibility](docs/compatibility.md) for the current differences,
classified as missing functionality, API/representation differences, unresolved
mismatches, or native Metal constraints. It also maps existing tests and future
compatibility checks for default streams, nesting, recapture, synchronization,
failures, and events. Missing functionality is not automatically a Metal limitation.

## How it works

1. Allocate buffers and compile handwritten MSL kernels.
2. Record dispatches, copies, and fills from a stream into a `Graph`.
3. Prepare a reusable backend plan, then launch it repeatedly.
4. Update bindings or constants transactionally when the workload changes.

Graphs retain buffers and kernels. Each submitted version keeps its resources
alive until completion, so later graph edits do not overwrite in-flight work.
The submission policy selects how the recorded commands reach Metal:

| Policy | Execution |
| --- | --- |
| `PerKernel` | Separate command buffers per command; useful as a baseline. |
| `Direct` | Encode the sequence into a single command buffer per submission. |
| `Indirect` | Reuse compute indirect command buffers for static compute segments; transfers and dynamic dispatches use separate encoding paths. Requires device support. |

Replay still involves CPU submission. Indirect command buffers reduce repeated
encoding; they do not turn the whole graph into a native CUDA graph equivalent.
Borrowed command buffers follow the external owner's submission boundaries.

## Build and run

Requires CMake 3.20+, C++17, and, for native execution, macOS 13+, an Apple SDK,
and a Metal-capable GPU.

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
./build/dependent_kernels --policy indirect
```

For CPU validation, including on other platforms:

```sh
cmake -S . -B build-reference -DMETALRT_ENABLE_METAL=OFF
cmake --build build-reference -j
ctest --test-dir build-reference --output-on-failure
./build-reference/dependent_kernels --reference
```

The reference backend runs explicit C++ callbacks; it does not interpret MSL or
measure GPU performance. Metal selection never silently falls back to the CPU.
Build switches are `METALRT_ENABLE_METAL`, `METALRT_BUILD_TESTS`,
`METALRT_BUILD_EXAMPLES`, `METALRT_TEST_METAL`, and `METALRT_SANITIZERS`.

## Explicit recording and replay

This complete example loads the supplied [affine shader](kernels/file_dispatch.metal).
Run it from the repository root so the shader path resolves.

```cpp
#include <metalrt/runtime.h>

int main() {
    using namespace metalrt;
    auto device = Device::system_default();
    Stream stream{device, DispatchPolicy::Indirect};
    auto kernel = device.kernel_file("kernels/file_dispatch.metal", "file_affine");
    constexpr uint32_t count = 256;
    auto input = device.allocate(count * sizeof(float));
    auto output = device.allocate(count * sizeof(float));
    for (uint32_t i = 0; i < count; ++i)
        input.data<float>()[i] = float(i);

    struct Params { uint32_t count; float scale, bias; };
    Graph graph;
    {
        auto recording = graph.record_guard(&stream);
        stream.dispatch(kernel, {count, 1, 1}, {128, 1, 1}, {
            Argument::buffer(0, input.view(), Access::Read),
            Argument::buffer(1, output.view(), Access::Write),
            Argument::scalar(2, Params{count, 2.0f, 1.0f})
        });
        recording.finish(); // Report recording errors here.
    }
    graph.prepare(&stream); // Optional: prepare without executing.
    graph.launch(&stream).wait();

    auto update = graph.update();
    update.set_scalar(0, 2, Params{count, 3.0f, 0.0f});
    update.commit();
    graph.launch(&stream).wait();
}
```

`record_guard()` records without executing; `capture_guard()` also submits on
successful completion. Explicit `finish()` reports errors immediately; destructor
errors are deferred to the stream. Exceptions cancel recording scopes. Host
reads and writes require completion of conflicting GPU work: `contents()` and
`data()` never synchronize. Scalar arguments are copied when recorded; buffer
contents remain live storage.

To embed the library, use `add_subdirectory(path/to/metalrt)` and link
`metalrt::metalrt`. Installed builds support `find_package(metalrt CONFIG REQUIRED)`
with the same target. Shader paths belong to the application; the library does
not embed a path to this repository.

## Beyond capture

| Capability | Code to start with |
| --- | --- |
| Update bindings/constants, clone graphs, retain in-flight versions | [replay_updates.cpp](examples/replay_updates.cpp) |
| Checked independent compute groups | [parallel_branches.cpp](examples/parallel_branches.cpp) |
| GPU predicates, bounded loops, GPU-produced dispatch counts | [gpu_control.cpp](examples/gpu_control.cpp), [control.h](include/metalrt/control.h) |
| Cross-stream timeline events and waits | [stream_events.cpp](examples/stream_events.cpp) |
| Reuse host/GPU ring slots with tracked completion | [ring_pipeline.cpp](examples/ring_pipeline.cpp), [ring.h](include/metalrt/ring.h) |
| Change typed parameters while keeping graph bindings stable | [stable_parameters.cpp](examples/stable_parameters.cpp), [parameters.h](include/metalrt/parameters.h) |
| Import native devices/buffers and borrow command buffers | [interop.h](include/metalrt/interop.h), [test_metal_interop.mm](tests/test_metal_interop.mm) |
| Specialize a common function, load offline libraries, sample kernel timestamps/counters | [native kernel guide](docs/native-kernels.md), [specialized_kernels.cpp](examples/specialized_kernels.cpp) |
| Retain an existing native compute pipeline | [imported_pipeline.mm](examples/imported_pipeline.mm) |
| Compare submission policies and completed GPU timing | [overhead_benchmark.cpp](examples/overhead_benchmark.cpp) |
| Upload images, sample with hardware filtering, and feed tiny-metal-nn training | [image sampling and training](docs/image-sampling.md), [texture.h](include/metalrt/texture.h) |

`Device::kernel_cached()` adds persistent Metal binary archives to the per-device
memory cache. Cache failures fall back to compilation and are counted in cache
statistics. Source-based archive hits still need a Metal library; use
`kernel_metallib()` to load offline-compiled code or `import_metal_pipeline()`
to retain a pipeline created by an external owner. `KernelCompileOptions::constants`
specializes a shared library function with Metal function constants.

`begin_gpu_timing(options)` optionally samples individual compute passes and
hardware counters when the device advertises support. `GpuTiming::occupied_seconds`
counts overlapping command-buffer intervals once; `command_buffer_seconds`
retains its original sum semantics. See the [native kernel guide](docs/native-kernels.md)
for profiling overhead and counter availability.

Structured GPU loops record a bounded number of iterations; they
are not unbounded device-side loops. Parallel groups validate declared buffer
accesses, so access annotations must reflect what the shader actually does.

`Device::upload_texture()` creates a retained, immutable 2D image in private GPU
storage. `Argument::texture()` and `Argument::sampler()` bind ordinary MSL
`texture2d<float>` and `sampler` parameters, with nearest/bilinear filtering and
clamp/repeat addressing. Sampled results can flow directly into training buffers.
The [image guide](docs/image-sampling.md) includes runnable sampling benchmarks,
an optional real tiny-metal-nn training example, and measured M3 Max results.

## Code layout and API direction

- `include/metalrt/`: focused public headers; `runtime.h` remains the core umbrella.
- `src/runtime/`: resources, validation, recording scopes, ordering, graph replay,
  and transactional updates. Shared implementation state stays private here.
- `src/backends/metal/`: native device/cache, plan preparation, submission,
  direct encoding, and indirect command buffer encoding.
- `src/backends/reference/`: deterministic CPU execution for semantic validation.
- `src/helpers/`: GPU control flow and ring storage built on the core API.
- `kernels/`, `examples/`, `tests/`: handwritten shaders, runnable use cases,
  and correctness/lifetime/package checks.

The guard-based capture interface remains the primary porting contract. Future
API work should make its behavioral compatibility precise and preserve that
calling pattern. Explicit recording/replay and the other helpers extend the
core while sharing its ownership, recording, and synchronization rules. The
[compatibility test priorities](docs/compatibility.md#compatibility-test-priorities)
distinguish existing coverage from checks still needed.
