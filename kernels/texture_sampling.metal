#include <metal_stdlib>
using namespace metal;

kernel void sample_image(texture2d<float, access::sample> image [[texture(0)]],
                         sampler sampling [[sampler(0)]],
                         device const float2* uv [[buffer(0)]],
                         device float4* rgb [[buffer(1)]],
                         constant uint& count [[buffer(2)]],
                         uint i [[thread_position_in_grid]]) {
    if (i < count) rgb[i] = image.sample(sampling, uv[i], level(0.0f));
}
kernel void consume_samples(device const float4* rgb [[buffer(0)]],
                            device float* sum [[buffer(1)]],
                            constant uint& count [[buffer(2)]],
                            uint i [[thread_position_in_grid]]) {
    if (i < count) sum[i] = dot(rgb[i], float4(1));
}

// Same coordinates, format, clamp-to-edge and bilinear rule as sample_image.
// This is a measurement baseline, not the new runtime texture implementation.
struct ImageDimensions { uint width, height, count; };
float4 buffer_texel(device const uchar4* pixels, int2 position, constant ImageDimensions& d) {
    const uint2 p = uint2(clamp(position,int2(0),int2(d.width-1,d.height-1)));
    return float4(pixels[p.y*d.width+p.x]) / 255.0f;
}
kernel void sample_image_buffer(device const uchar4* pixels [[buffer(0)]],
                                 device const float2* uv [[buffer(1)]],
                                 device float4* rgb [[buffer(2)]],
                                 constant ImageDimensions& d [[buffer(3)]],
                                 uint i [[thread_position_in_grid]]) {
    if (i >= d.count) return;
    float2 p = clamp(uv[i],0.0f,1.0f)*float2(d.width,d.height)-.5f;
    int2 lo = int2(floor(p));
    float2 f = fract(p);
    rgb[i] = mix(mix(buffer_texel(pixels,lo,d),buffer_texel(pixels,lo+int2(1,0),d),f.x),
                 mix(buffer_texel(pixels,lo+int2(0,1),d),buffer_texel(pixels,lo+int2(1,1),d),f.x),f.y);
}
