#include <metal_stdlib>
using namespace metal;
uint image_hash(uint x) {
    x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu;
    return x ^ (x >> 16);
}
struct SamplingParams { uint count, seed; };
kernel void sample_training_image(texture2d<float, access::sample> image [[texture(0)]],
                                   sampler filtering [[sampler(0)]],
                                   device float* inputs [[buffer(0)]],
                                   device float* targets [[buffer(1)]],
                                   constant SamplingParams& p [[buffer(2)]],
                                   uint i [[thread_position_in_grid]]) {
    if (i >= p.count) return;
    const float2 uv = float2(image_hash(i*2u+p.seed)>>8,
                            image_hash(i*2u+1u+p.seed)>>8)*0x1p-24f;
    const float3 rgb = image.sample(filtering,uv,level(0.0f)).rgb;
    // GPUMatrix<float> column-major matrices: (2,N) and (3,N).
    // Scalar stores avoid Metal float3's 16-byte alignment/padding.
    inputs[2*i] = uv.x; inputs[2*i+1] = uv.y;
    targets[3*i] = rgb.r; targets[3*i+1] = rgb.g; targets[3*i+2] = rgb.b;
}
struct EvaluationParams { uint width, height; };
kernel void sample_evaluation_image(texture2d<float, access::sample> image [[texture(0)]],
                                     sampler filtering [[sampler(0)]],
                                     device float* inputs [[buffer(0)]],
                                     device float* targets [[buffer(1)]],
                                     constant EvaluationParams& p [[buffer(2)]],
                                     uint i [[thread_position_in_grid]]) {
    if (i >= p.width*p.height) return;
    const float2 uv = (float2(i%p.width,i/p.width)+.5f)/float2(p.width,p.height);
    const float3 rgb = image.sample(filtering,uv,level(0.0f)).rgb;
    inputs[2*i] = uv.x; inputs[2*i+1] = uv.y;
    targets[3*i] = rgb.r; targets[3*i+1] = rgb.g; targets[3*i+2] = rgb.b;
}
