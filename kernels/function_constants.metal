#include <metal_stdlib>
using namespace metal;
constant float scale [[function_constant(0)]];
constant uint value_bias [[function_constant(1)]];
constant bool enabled [[function_constant(2)]];
constant int offset [[function_constant(3)]];
kernel void specialized_value(device float* output [[buffer(0)]], uint i [[thread_position_in_grid]]) {
    output[i] = enabled ? float(i) * scale + float(value_bias) + float(offset) : 0.0f;
}
