#ifndef _GALAXY_KERNEL_H_
#define _GALAXY_KERNEL_H_

#include "operatorsCuda.h"
#include "wing.h"

#define BSIZE 256
#define softeningSquared 0.01f		// original plumer softener is 0.025. here the value is square of it.
#define damping 1.0f				// 0.999f
#define ep 0.67f					// 0.5f
#define box 100						// box size
#define particleRadius 0.05f
#define particleMass 1.f

extern Wing* h_wing;
extern Wing* d_wing;

__device__ float3 bodyBodyInteraction(float4& particle1, float4& myVelocity, float4 particle2, float3 ai)
{
	float3 p1 = make_float3(particle1.x, particle1.y, particle1.z);
	float3 p2 = make_float3(particle2.x, particle2.y, particle2.z);

	float distance = length(p2 - p1);
	float radius = particleRadius + particleRadius;

	if(distance <= radius)
	{
		// myVelocity.x = myVelocity.x * (-1);
		// myVelocity.y = myVelocity.y * (-1);
		// myVelocity.z = myVelocity.z * (-1);

		// float zaDaleko = ((radius - distance) / 2) + 0.01f; // odsuwam troche dalej zeby nie bylo kolejnej kolizji
		
		// // float3 velocityTmp = make_float3(myVelocity.x, myVelocity.y, myVelocity.z);
		// float3 velocityTmp;
		// velocityTmp.x = myVelocity.x;
		// velocityTmp.y = myVelocity.y;
		// velocityTmp.z = myVelocity.z;

		// velocityTmp = velocityTmp * (
		// 	1);
		// float3 odsun = versor(velocityTmp) * zaDaleko;

		// particle1 = make_float4(particle1.x + odsun.x,
		// 						particle1.y + odsun.y,
		// 						particle1.z + odsun.z,
		// 						0);

	}


	float3 wind = make_float3(50.0, -0.0, 0.0);
	ai = wind; 
    return ai;
}

__device__ float3 tile_calculation(float4& myPosition, float4& myVelocity, float3 acc)
{
	extern __shared__ float4 shPosition[];
	
	#pragma unroll 8
	for (unsigned int i = 0; i < BSIZE; i++)
		acc = bodyBodyInteraction(myPosition, myVelocity, shPosition[i], acc);
		
	return acc;
}

__device__ void boxCollision(float4& myPosition, float4& myVelocity)
{
	if(myPosition.x < -box/2)
	{
		myPosition.x += box;
		//myPosition.x = -box - (myPosition.x);
		//myVelocity.x = -myVelocity.x;
	}
	if(myPosition.x > box/2)
	{
		myPosition.x -= box;
		//myPosition.x = box - (myPosition.x);
		//myVelocity.x = -myVelocity.x;
	}	
	if(myPosition.y  < -box/2)
	{
		myPosition.y = -box - myPosition.y;
		myVelocity.y = -myVelocity.y;
	}
	if(myPosition.y > box/2)
	{
		myPosition.y = box - myPosition.y;
		myVelocity.y = -myVelocity.y;
	}			
	if(myPosition.z < -box/2)
	{
		myPosition.z = -box - myPosition.z;
		myVelocity.z = -myVelocity.z;
	}
	if(myPosition.z > box/2)
	{
		myPosition.z = box - myPosition.z;
		myVelocity.z = -myVelocity.z;
	}
}

__device__ void wingCollision(float4& myPosition, float4& myVelocity, float3 acc, Wing* wing)
{
	float aa_M_PI = 0.31830988618379067154;
	float e = 0.9f; // strata energi

	float3 force = make_float3(.0f, .0f, .0f);
		
	//gorny plat
	if(myPosition.x >= wing->pos.x && 
	   myPosition.x <= wing->pos.x + wing->length)
	{
		float x1 = wing->pos.x ;				//lewa gora
		float y1 = wing->pos.y + wing->radius;
		float x2 = wing->pos.x + wing->length;	//tyl
		float y2 = 0;
		float x3 = wing->pos.x;					//lewy dol
		float y3 = wing->pos.y - wing->radius;

		float test_y_gora = (y2 - y1)*(myPosition.x - x1) / (x2 - x1) + y1;	//wysokosc skrzydla dla danego x 
		float test_y_dol = (y2 - y3)*(myPosition.x - x3) / (x2 - x3) + y3;	//wysokosc skrzydla dla danego x 
		
		//gormy plat
		if(myPosition.y + particleRadius <= test_y_gora && 
		   myPosition.y + particleRadius > test_y_dol)	
		{
			myPosition.y = test_y_gora + particleRadius;
			
			float kat = atan(wing->radius/wing->length) * 180 / aa_M_PI;
			myVelocity.x += myVelocity.y * sin(kat);
			myVelocity.y = -myVelocity.y * e;	

			//wing->sila_nosna = dodaj(wing->sila_nosna, tab[idx].f);
			wing->sila_nosna = wing->sila_nosna + acc;

		}
		//dolny plat
		if(myPosition.y + particleRadius >= test_y_dol && 
		   myPosition.y + particleRadius < test_y_gora )		
		{
			myPosition.y = test_y_dol - particleRadius;
			myVelocity.y = -myVelocity.y * e;	

			//wing->sila_nosna = dodaj(wing->sila_nosna, tab[idx].f);
			wing->sila_nosna = wing->sila_nosna + acc;
		}
	}
	

	// walec
	float3 wingPos = make_float3(wing->pos.x, wing->pos.y, myPosition.z); // teraz czaska jest na przeciwko szkrzydla
	float distance = length(myPosition - wingPos);
	float radius = particleRadius + wing->radius;

	if(distance <= radius)
	{	
		myVelocity = myVelocity * (-1);
		wing->sila_nosna = wing->sila_nosna + acc;
		
		// is that rly needed?
		// float tooFar = radius - distance;
		// float3 myPosTmp = make_float3(myPosition.x, myPosition.y, myPosition.z);
		// normalize(myPosTmp);


		// //myPosTmp = myPosTmp + (myPosTmp * tooFar);
		// myPosition = make_float4(myPosTmp.x, myPosTmp.y, myPosTmp.z, 1);
	} 
}

__global__ void galaxyKernel(float4* pos, float4* pdata, unsigned int width, 
			 				 unsigned int height, float step, int apprx, int offset,
			 				 Wing* wing)
{
	// shared memory
	extern __shared__ float4 shPosition[];
	
	// index of my body	
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int pLoc = y * width + x;
    unsigned int vLoc = width * height + pLoc;
	
    // starting index of position array
    unsigned int start = ( (width * height) / apprx ) * offset;
	
	float4 myPosition = pdata[pLoc];
	float4 myVelocity = pdata[vLoc];

	float3 acc = {0.0f, 0.0f, 0.0f};

	unsigned int idx = 0;
	unsigned int loop = ((width * height) / apprx ) / BSIZE;
	for (int i = 0; i < loop; i++)
	{
		idx = threadIdx.y * blockDim.x + threadIdx.x;
		shPosition[idx] = pdata[idx + start + BSIZE * i];

		__syncthreads();
		
		acc = tile_calculation(myPosition, myVelocity, acc);
		
		__syncthreads();		
	}
    	
    // update velocity with above acc
    myVelocity.x += acc.x * step;// * 2.0f;
    myVelocity.y += acc.y * step;// * 2.0f;
    myVelocity.z += acc.z * step;// * 2.0f;
    
    myVelocity.x *= damping;
    myVelocity.y *= damping;
    myVelocity.z *= damping;
    
    // update position
    myPosition.x += myVelocity.x * step;
    myPosition.y += myVelocity.y * step;
    myPosition.z += myVelocity.z * step;
        
	boxCollision(myPosition, myVelocity);
	wingCollision(myPosition, myVelocity, acc, wing);

    __syncthreads();
    
    // update device memory
	pdata[pLoc] = myPosition;
	pdata[vLoc] = myVelocity;
    
	// update vbo
	pos[pLoc] = make_float4(myPosition.x, myPosition.y, myPosition.z, 1.0f);
	pos[vLoc] = myVelocity;

}

extern "C" 
void cudaComputeGalaxy(float4* pos, float4 * pdata, int width, int height, 
					   float step, int apprx, int offset, Wing* wing)
{
    dim3 block(16, 16, 1);
    dim3 grid(width / block.x, height / block.y, 1);
    int sharedMemSize = BSIZE * sizeof(float4);
    galaxyKernel <<< grid, block, sharedMemSize >>> (pos, pdata, width, height, 
    											 	 step, apprx, offset, wing);
}

#endif