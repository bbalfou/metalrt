#include <metal_stdlib>
using namespace metal;

struct Parameters { uint count; float scale, bias; uint pad; };
kernel void stable_affine(device const float* input [[buffer(0)]],
                          device float* output [[buffer(1)]],
                          constant Parameters& parameters [[buffer(2)]],
                          uint i [[thread_position_in_grid]]) {
    if(i<parameters.count) output[i]=input[i]*parameters.scale+parameters.bias;
}
