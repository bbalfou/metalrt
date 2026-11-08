#include <metal_stdlib>
using namespace metal;

struct WriteConfig { uint capacity, width, height, value; };
struct GroupCounts { uint x,y,z; };

kernel void control_write(device uint* output [[buffer(0)]],
                          constant WriteConfig& config [[buffer(1)]],
                          uint3 i [[thread_position_in_grid]]) {
    uint linear=i.x+config.width*(i.y+config.height*i.z);
    if(linear<config.capacity)output[linear]=config.value+linear;
}
kernel void control_store(device uint* output [[buffer(0)]],
                          constant uint& value [[buffer(1)]],uint i [[thread_position_in_grid]]) {
    if(i==0)output[0]=value;
}
kernel void control_increment(device uint* value [[buffer(0)]],
                              constant uint& amount [[buffer(1)]],uint i [[thread_position_in_grid]]) {
    if(i==0)value[0]+=amount;
}
kernel void control_decrement(device uint* value [[buffer(0)]],uint i [[thread_position_in_grid]]) {
    if(i==0 && value[0]!=0)--value[0];
}
kernel void control_counts(device uint* counts [[buffer(0)]],
                           constant GroupCounts& value [[buffer(1)]],uint i [[thread_position_in_grid]]) {
    if(i==0){counts[0]=value.x;counts[1]=value.y;counts[2]=value.z;}
}
kernel void control_atomic_add(device atomic_uint* counter [[buffer(0)]]) {
    atomic_fetch_add_explicit(counter,1u,memory_order_relaxed);
}
