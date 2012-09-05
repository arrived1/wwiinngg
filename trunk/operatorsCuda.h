#ifndef OPERATORS_CUDA_HPP
#define OPERATORS_CUDA_HPP

#include <cuda.h>

__device__ float3 operator+(const float3& a, const float3& b)
{
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ float3 operator-(const float3& a, const float3& b)
{
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ float3 operator%(const float3& a, const float3& b)
{
    return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}

__device__ float operator*(const float3& a, const float3& b)
{
    return (a.x * b.x + a.y * b.y + a.z * b.z);
}

__device__ float3 operator*(const float3& a, float number)
{
        return make_float3(a.x * number, a.y * number, a.z * number);
}

__device__ float3 operator*(float number, const float3& a)
{
        return make_float3(a.x * number, a.y * number, a.z * number);
}

__device__ float length(const float3& a)
{
    return sqrt(a*a);
}

__device__ void normalize(float3& a)
{
    float d = length(a);
    if(d)
        a = a * (1 / d);
}


//hacks
__device__ float3 operator-(const float4& a, const float3& b)
{
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

// __device__ float3 operator+(const float4& a, const float3& b)
// {
//     return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
// }

__device__ float4 operator+(const float4& a, const float3& b) // is this ok?
{
    return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, 0);
}


#endif // OPERATORS_CUDA_HPP
