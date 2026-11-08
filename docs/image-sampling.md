# Image sampling and tiny-metal-nn training

The runtime supports immutable, single-level 2D images and sampler arguments.
Metal uploads pixels once through a staging buffer into a private, shader-read
`MTLTexture`. This gives the sampler an optimized image layout and hardware
bilinear filtering. It is a real 2D texture, rather than a linear `texture_buffer`
or a renamed `MTLBuffer`. The image codec is example-only; the runtime accepts
pixels and has no image decoding dependency.

## Build and demonstrate

From the metalrt repository root on macOS:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/image_sampling --image data/images/Albert_Einstein_Head.jpg
```

The example decodes the supplied JPEG with macOS ImageIO, uploads its full
3250 × 4333 RGBA image, and samples 1,048,576 random normalized coordinates per
batch. It checks every result against a manual bilinear buffer kernel, checks
4,096 samples against the CPU reference implementation, then measures 50 graph
replays of each kernel with completed Metal command-buffer timestamps. The
hardware and buffer kernels use the same pixels, UVs, clamp rule, and float4
output layout. Small differences from hardware interpolation precision are
expected; the demonstration rejects differences greater than 0.006.

`--samples N` and `--iterations N` change the workload. `--indirect` exercises
the mixed direct/indirect graph execution path. Run from the repository root
or supply an explicit image path. Metal errors are reported without CPU fallback.

## Public API

```cpp
// rgba contains width*height*4 bytes, in top-to-bottom row order.
auto image = device.upload_texture(
    {width, height, metalrt::PixelFormat::RGBA8Unorm}, rgba.data(), rgba.size());
auto sampler = device.sampler(); // Bilinear, normalized coordinates, clamp to edge.

stream.dispatch(kernel, {count, 1, 1}, {128, 1, 1}, {
    metalrt::Argument::texture(0, image),
    metalrt::Argument::sampler(0, sampler),
    metalrt::Argument::buffer(0, coordinates.view()),
    metalrt::Argument::buffer(1, colors.view(), metalrt::Access::Write),
    metalrt::Argument::scalar(2, count) // uint32_t, matching the shader
});
```

The matching MSL signature and sample operation are in
[`texture_sampling.metal`](../kernels/texture_sampling.metal):

```metal
kernel void sample_image(texture2d<float, access::sample> image [[texture(0)]],
                         sampler sampling [[sampler(0)]],
                         device const float2* uv [[buffer(0)]],
                         device float4* rgb [[buffer(1)]],
                         constant uint& count [[buffer(2)]],
                         uint i [[thread_position_in_grid]]) {
    if (i < count) rgb[i] = image.sample(sampling, uv[i], level(0.0f));
}
```

Texture, sampler, buffer/constant, and threadgroup indices have independent
namespaces. In particular, binding texture(0), sampler(0), and buffer(0) together
is valid. The portable slot ranges are 0–30 for textures and buffers and 0–15
for samplers. The upload accepts an optional host byte stride and a source size,
and rejects undersized images, short strides, invalid formats, and dimensions
outside 1–16384. It snapshots host bytes and completes the upload before returning,
so the caller can immediately free the input pixels or sample on another stream.
Upload is a synchronous setup operation; it is not recorded in a graph.

Supported formats are `R8Unorm`, `RGBA8Unorm`, `R32Float`, and `RGBA32Float`.
The single-channel formats sample as `(r, 0, 0, 1)`. Coordinates are normalized;
texel centers are `((x+.5)/width, (y+.5)/height)`. Row zero corresponds to `v=0`.
`SamplerDescriptor` selects `Filter::Nearest` or `Filter::Linear`, with independent
`AddressMode::ClampToEdge` or `AddressMode::Repeat` for U and V. There is one mip
level, selected explicitly with `level(0.0f)` in compute shaders. Pixels have no
implicit sRGB transfer function: RGBA8 values are divided by 255. The examples
train on those display-space values.

Handles are retained by recorded commands, prepared plans, and in-flight
submissions. `Graph::Update::set_texture()` and `set_sampler()` replace bindings
transactionally while older launches retain their original bindings. Texture
contents are immutable; upload a new image to change them. Private textures have
no host pointer. `Texture2D::reference_sample()` is available only to explicit
CPU reference callbacks for validation.

With `DispatchPolicy::Indirect`, texture/sampler commands are encoded directly
between reusable buffer-only ICB segments. They remain in the same submission
and retain normal producer/consumer ordering. This version does not lower
texture/sampler arguments to Metal argument buffers inside an ICB.

Apple documents [private resource uploads](https://developer.apple.com/documentation/metal/copying-data-to-a-private-resource),
[normalized sampler coordinates](https://developer.apple.com/documentation/metal/mtlsamplerdescriptor/normalizedcoordinates),
and the [indirect compute buffer table](https://developer.apple.com/documentation/metal/mtlindirectcomputecommand/setkernelbuffer(_:offset:at:)).

## Train the actual tiny-metal-nn MLP

An optional example links a separate tiny-metal-nn checkout against this build's
metalrt target. Its bundled runtime is bypassed; its checkout is unchanged.
The tiny-metal-nn checkout needs its normal dependencies initialized.

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DMETALRT_TINY_METAL_NN_SOURCE_DIR="$PWD/../tiny-metal-nn"
cmake --build build -j
./build/image_training --image data/images/Albert_Einstein_Head.jpg \
  --steps 1000 --batch-size 16384 --out build/image-training
```

[`image_training.cpp`](../examples/image_training.cpp) uses the actual
`tcnn::NetworkWithInputEncoding`, `Trainer`, and Adam optimizer, with a 12-level
HashGrid encoding and a 64-neuron FullyFusedMLP with two hidden layers. The
[`sampling shader`](../kernels/image_training.metal) generates new random UVs
each step and samples the texture into the network's existing
`GPUMatrix<float>` allocations: `(2, N)` coordinates and `(3, N)` RGB targets in
column-major layout. RGB uses three scalar stores per sample because MSL
`float3` has 16-byte alignment. This is the integration point for other networks
that consume the same training layout.

Sampling, encoding, forward/backward, and optimization share one runtime stream
and one batch. No image upload or CPU target generation occurs per step. Scalar
seed constants change on the host; progress reads loss only every 100 steps and
at completion. Batch size must be a positive multiple of 128. Add `--indirect`
to exercise mixed sampling and indirect NN commands.

The example evaluates before and after training on a fixed, independent 256 × 341
coordinate grid for this image, and fails if error does not decrease or values
become non-finite. It writes `target.png`, `prediction.png`, and `results.json`
under the requested directory. The target is the texture sampled on that
evaluation grid; reported PSNR compares unclipped predictions against those
targets, rather than against every original-resolution source pixel.

## Measured demonstration

On the local 40-core Apple M3 Max, Release build, supplied JPEG, with Metal
instrumentation disabled, a representative sampling run produced:

| Kernel | GPU ms per 1,048,576 samples | Million samples/s |
| --- | ---: | ---: |
| Manual bilinear RGBA8 buffer | 0.440 | 2381 |
| Hardware bilinear RGBA8 texture | 0.140 | 7492 |

That is approximately **3.15×** for this workload. Maximum per-channel difference
was about 0.001. Uploading the 56,329,000 source bytes took 34 ms in that process;
other local runs took 11–30 ms. Upload, JPEG decoding, coordinate generation,
compilation, and warmup are excluded from sampling timing. The benchmark reuses
the same UV batch and image; throughput depends on GPU, image, locality, cache
state, and instrumentation. GPU times are command-buffer intervals, not an
exclusive measure of sampler utilization.

The 1,000-step tiny-metal-nn run used batch size 16,384 and 1,325,904 parameters:

| Fixed evaluation metric | Before | After |
| --- | ---: | ---: |
| MSE | 0.222258 | 0.000254 |
| PSNR | 6.53 dB | 35.96 dB |

This demonstrates upload → texture sampling → real network training, rather
than just allocating an image or timing an isolated shader. Exact training
results and timing vary between runs; `results.json` records the local result.

## Validation

```sh
ctest --test-dir build --output-on-failure
MTL_DEBUG_LAYER=1 MTL_SHADER_VALIDATION=1 ./build/test_texture --metal
```

The texture suite runs on Reference and Metal. It covers all four formats,
odd widths, padded row strides, copied upload ownership, nearest/bilinear
filtering, repeat/clamp addressing, texel centers and edges, independent argument
namespaces (including optional buffers and derived pointer bases), foreign-device
rejection, invalid bindings/uploads, graph preparation/replay and transactional
edits, in-flight version retention, mixed ICB dependencies, and GPU-driven grids.
The original runtime, interop, and installed-consumer checks also remain enabled.
