// Shared by the graph-update, parallel-branch, and cross-stream examples/tests.
#include <metal_stdlib>
using namespace metal;

struct GraphAffineArgs {uint count;float scale,bias;};
kernel void graph_affine(device const float* input [[buffer(0)]],
                         device float* output [[buffer(1)]],
                         constant GraphAffineArgs& args [[buffer(2)]],
                         uint i [[thread_position_in_grid]]) {
    if(i<args.count)output[i]=input[i]*args.scale+args.bias;
}

kernel void graph_sum(device const float* left [[buffer(0)]],
                      device const float* right [[buffer(1)]],
                      device float* output [[buffer(2)]],
                      constant uint& count [[buffer(3)]],
                      uint i [[thread_position_in_grid]]) {
    if(i<count)output[i]=left[i]+right[i];
}
