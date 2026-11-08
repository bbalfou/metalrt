// Handwritten Metal Shading Language; loaded by Device::kernel_file.
#include <metal_stdlib>
using namespace metal;
struct Shape { uint count; };
kernel void encode_features(device const float* input [[buffer(0)]],
                            device float* encoded [[buffer(1)]],
                            constant Shape& shape [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if(i>=shape.count) return;
    float x=input[2*i], y=input[2*i+1];
    encoded[4*i]=x; encoded[4*i+1]=y;
    encoded[4*i+2]=sin(3.14159265358979323846f*x);
    encoded[4*i+3]=cos(3.14159265358979323846f*y);
}
kernel void dense_layer(device const float* encoded [[buffer(0)]],
                        device const float* weights [[buffer(1)]],
                        device float* hidden [[buffer(2)]],
                        constant Shape& shape [[buffer(3)]], uint i [[thread_position_in_grid]]) {
    if(i>=shape.count*3) return;
    uint sample=i/3, neuron=i%3;
    float value=0;
    for(uint j=0;j<4;++j) value+=weights[neuron*4+j]*encoded[sample*4+j];
    hidden[i]=value;
}
kernel void relu_reduce(device const float* hidden [[buffer(0)]],
                        device float* output [[buffer(1)]],
                        constant Shape& shape [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if(i>=shape.count) return;
    output[i]=max(0.0f,hidden[3*i])+max(0.0f,hidden[3*i+1])+max(0.0f,hidden[3*i+2]);
}
