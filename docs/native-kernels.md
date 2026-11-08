# Native kernels and GPU measurements

The runtime now supports Metal function constants, offline libraries, retained
native pipelines, and opt-in compute-pass counter sampling. These fill API gaps:
previously source compilation called `newFunctionWithName:` without specialization,
all user libraries came from `newLibraryWithSource:`, and profiling collected only
completed command-buffer timestamps. The runtime itself did not contain a
`#define` specialization implementation; applications generating source can now
replace that pattern with function constants.

## Function constants

Declare scalar constants in ordinary MSL:

```metal
constant float scale [[function_constant(0)]];
kernel void affine(device float* out [[buffer(0)]], uint i [[thread_position_in_grid]]) {
    out[i] = float(i) * scale;
}
```

Supply typed values by index:

```cpp
metalrt::KernelCompileOptions options;
options.constants = {metalrt::FunctionConstant::scalar(0, 2.0f)};
auto twice = device.kernel(source, "affine", options);
options.constants = {metalrt::FunctionConstant::scalar(0, 5.0f)};
auto five_times = device.kernel(source, "affine", options);
```

Supported types are `bool`, `int32_t`, `uint32_t`, and `float`. Indices and types
must match the shader; required constants must be supplied. Invalid/duplicate
indices and malformed payloads throw. Input order does not affect cache identity.
Source libraries are shared across variants with the same compile options;
each specialized direct/indirect pipeline has its own memory and archive key.
`kernel_cached(source, entry, directory, options)` also persists specialized
pipeline archives. Archives alone do not eliminate source-to-IR compilation in a
new process. Recorded/in-flight commands retain variants across cache clearing.

Reference callbacks implement their own specialized semantics, for example by
capturing the same scale. The Reference backend never interprets MSL constants.

## Offline libraries and existing pipelines

```cpp
auto kernel = device.kernel_metallib("my_kernels.metallib", "affine", options);
```

This loads compiled library bytes with `newLibraryWithData:` and specializes the
function without compiling MSL. GPU pipeline creation still occurs. The file
must be a regular, nonempty Metal library; load errors throw without falling
back to source. The cache keys use file contents, so replacing a library at the
same path cannot return stale code. `fast_math` applies to source compilation;
for offline libraries that choice is already baked into the file.

The build creates `build/shaders/function_constants.metallib` when Apple's
optional Metal compiler toolchain is installed:

```sh
./build/specialized_kernels
./build/specialized_kernels --reference
./build/specialized_kernels --metallib build/shaders/function_constants.metallib
./build/imported_pipeline
```

For application-owned pipelines, include `<metalrt/interop.h>` and pass borrowed
Objective-C handles from Objective-C++:

```cpp
auto kernel = metalrt::import_metal_pipeline(device, (__bridge void*)direct,
                                            "affine", (__bridge void*)indirect);
```

The runtime retains both objects and validates their devices. The optional
indirect pipeline must support indirect command buffers and have the same ABI
and behavior as the direct pipeline. Without it, use Direct or PerKernel policy.
No reflection is available for imported pipelines: callers are responsible for
correct resource bindings; null-buffer reflection checks are unavailable.
Pipeline import does not compile, rebuild, or persist the pipeline.

## Kernel sampling and counters

```cpp
metalrt::GpuTimingOptions options;
options.sample_kernels = true;
// Optional; check capabilities().statistic_counters before requesting this.
options.counters = metalrt::GpuCounterSet::Statistic;
stream.begin_gpu_timing(options);
graph.launch(&stream);
auto timing = stream.end_gpu_timing();
for (const auto& kernel : timing.kernels) {
    if (kernel.valid) std::cout << kernel.kernel << ": " << kernel.seconds << " s\n";
    for (const auto& counter : kernel.counters)
        if (counter.valid()) std::cout << counter.name << ": " << counter.begin << " -> " << counter.end << '\n';
}
```

`Capabilities::kernel_timestamps`, `statistic_counters`, and `utilization_counters`
report encoder-boundary support and counter-set availability. Requests for
unsupported sampling throw before opening a window. Ordinary command-buffer
timing remains available without counters. Reference execution returns no GPU
measurements and rejects counter-sampling requests.

Samples identify the entry name and command index within their submission.
They retain raw GPU timestamps; durations use paired CPU/GPU clock calibrations
before encoding and after completion. Hardware counter results expose the
standard Metal Statistic or StageUtilization structure fields as raw beginning
and ending values. Missing fields retain Metal's `UINT64_MAX` error sentinel;
check `valid()` before using them. Counter meanings and availability vary by
GPU. These values do not promise exclusive activity attributable to one kernel.

Sampling uses a separate serial compute pass for each dispatch, including
Indirect-policy replays. This changes overlap, adds pass boundaries and sample
allocation overhead, and temporarily prepares a Direct plan. Samples are for
instrumented execution, not a measurement of the original concurrent schedule.
Dynamic dispatch samples cover the payload pass, excluding its argument
preparation kernel. Transfers are included in command-buffer timing but do not
produce kernel samples. Empty windows contain no samples. Repeated ticket waits
collect samples once. Borrowed-stream owners still need to commit before waits.

`command_buffer_seconds` retains the sum of completed command-buffer intervals,
which can double-count overlap. `occupied_seconds` is their union, counting
overlap once while excluding gaps. `span_seconds()` includes gaps. None of these
is exclusive GPU ALU busy time. These meanings also apply without kernel sampling.

The sampling implementation follows Apple's [counter sampling API](https://developer.apple.com/documentation/metal/sampling-gpu-data-into-counter-sample-buffers)
and [counter formats and clock guidance](https://developer.apple.com/documentation/metal/gpu-counters-and-counter-sample-buffers).
