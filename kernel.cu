#ifndef _KERNEL_H_
#define _KERNEL_H_

#include "operatorsCuda.h"
#include "wing.h"

#include <thrust/device_vector.h>

#define BSIZE 256
#define softeningSquared 0.01f		// original plumer softener is 0.025. here the value is square of it.
#define damping 1.0f				// 0.999f
#define ep 0.67f					// 0.5f
#define box 100						// box size
#define particleRadius 0.05f
#define particleMass 1.f

extern Wing* h_wing;
extern Wing* d_wing;

thrust::device_vector<float4> dpos;

extern "C" void runKernel(float4* pos, float4* pdata, int width, int height, float step, Wing* wing)
{
    // dim3 block(16, 16, 1);
    // dim3 grid(width / block.x, height / block.y, 1);
    // int sharedMemSize = BSIZE * sizeof(float4);
    // galaxyKernel <<< grid, block, sharedMemSize >>> (pos, pdata, width, height, 
    // 											 	 step, apprx, offset, wing);

}

#endif