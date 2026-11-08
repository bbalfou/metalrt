#include <metal_stdlib>
using namespace metal;

struct RingArgs { uint count, stride_floats; float scale, bias; };

// A fixed binding variant for kernels that expect one slot's base pointer.
kernel void ring_per_slot(device float* slot [[buffer(0)]],
                          constant RingArgs& args [[buffer(1)]],
                          uint i [[thread_position_in_grid]]) {
    if(i<args.count) slot[args.count+i]=slot[i]*args.scale+args.bias;
}

// One unchanged ICB can address any ring slot. The runtime advances this
// GPU-owned index in queue order. CPU code never rewrites the selector buffer.
kernel void ring_indexed(device float* storage [[buffer(0)]],
                         device const uint* active_slot [[buffer(1)]],
                         constant RingArgs& args [[buffer(2)]],
                         uint i [[thread_position_in_grid]]) {
    if(i>=args.count) return;
    uint base=active_slot[0]*args.stride_floats;
    storage[base+args.count+i]=storage[base+i]*args.scale+args.bias;
}
