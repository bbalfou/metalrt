// Handwritten Metal Shading Language; loaded by Device::kernel_file.
#include <metal_stdlib>
using namespace metal;
struct AffineParams { uint n; float scale; float bias; };
kernel void runtime_affine(device const float* input [[buffer(0)]],
                           device float* output [[buffer(1)]],
                           constant AffineParams& p [[buffer(2)]],
                           uint tid [[thread_position_in_grid]]) {
    if(tid<p.n) output[tid]=p.scale*input[tid]+p.bias;
}
kernel void runtime_square(device const float* input [[buffer(0)]],
                           device float* output [[buffer(1)]],
                           constant uint& n [[buffer(2)]],
                           uint tid [[thread_position_in_grid]]) {
    if(tid<n) output[tid]=input[tid]*input[tid];
}
kernel void runtime_threadgroup_exchange(device const float* input [[buffer(0)]],
                                         device float* output [[buffer(1)]],
                                         constant uint& n [[buffer(2)]],
                                         threadgroup float* scratch [[threadgroup(0)]],
                                         uint tid [[thread_position_in_grid]],
                                         uint lane [[thread_index_in_threadgroup]]) {
    scratch[lane]=tid<n ? input[tid] : 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(tid<n) output[tid]=scratch[lane^1u];
}
