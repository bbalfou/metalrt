#include <metal_stdlib>
using namespace metal;
kernel void consumer_twice(device float* value [[buffer(0)]],uint i [[thread_position_in_grid]]) {
    if(i==0)value[0]*=2.0f;
}
