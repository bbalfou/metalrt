// A small standalone file-loading example. No C++ source string is generated.
#include <metal_stdlib>
using namespace metal;

struct FileDispatchArgs {
    uint count;
    float scale;
    float bias;
};

kernel void file_affine(device const float* input [[buffer(0)]],
                        device float* output [[buffer(1)]],
                        constant FileDispatchArgs& args [[buffer(2)]],
                        uint index [[thread_position_in_grid]]) {
    if(index>=args.count) return;
    output[index]=input[index]*args.scale+args.bias;
}
