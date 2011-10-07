#ifndef KERNELGLOBALSCUDA_H
#define KERNELGLOBALSCUDA_H

#include "CUDAGlobals.cuh"

#include "../VectorUtils.cuh"
#include "../AgentGroupData.cuh"

#define THREADSPERBLOCK 256

#define EPSILON	0.001f

//#define TIMING

// Kernel bit masks.
static uint const KERNEL_PURSUE_BIT				= 1;
static uint const KERNEL_EVADE_BIT				= 1 << 1;
static uint const KERNEL_SEEK_BIT				= 1 << 2;
static uint const KERNEL_FLEE_BIT				= 1 << 3;

static uint const KERNEL_SEPARATION_BIT			= 1 << 4;
static uint const KERNEL_ALIGNMENT_BIT			= 1 << 5;
static uint const KERNEL_COHESION_BIT			= 1 << 6;

static uint const KERNEL_AVOID_OBSTACLES_BIT	= 1 << 7;
static uint const KERNEL_AVOID_WALLS_BIT		= 1 << 8;
static uint const KERNEL_AVOID_NEIGHBORS_BIT	= 1 << 9;

static uint const KERNEL_FOLLOW_PATH_BIT		= 1 << 10;

#define COALESCE
#if defined COALESCE
	#define FLOAT3_GLOBAL_READ( shDest, pdSource )		{																							\
																__syncthreads();																	\
																if(blockIdx.x < gridDim.x-1)														\
																{																					\
																	float3_coalescedRead( shDest, pdSource, blockIdx.x, blockDim.x, threadIdx.x );	\
																}																					\
																else																				\
																{																					\
																	shDest[threadIdx.x] = pdSource[(blockIdx.x * blockDim.x) + threadIdx.x];		\
																}																					\
																__syncthreads();																	\
															}
	#define FLOAT3_GLOBAL_WRITE( pdDest, shSource )		{																							\
																__syncthreads();																	\
																if(blockIdx.x < gridDim.x-1)														\
																{																					\
																	float3_CoalescedWrite( pdDest, shSource, blockIdx.x, blockDim.x, threadIdx.x );	\
																}																					\
																else																				\
																{																					\
																	pdDest[(blockIdx.x * blockDim.x) + threadIdx.x] = shSource[threadIdx.x];		\
																}																					\
																__syncthreads();																	\
															}
#else
	#define FLOAT3_GLOBAL_READ( shDest, pdSource )		{																							\
															__syncthreads();																		\
															shDest[threadIdx.x] = pdSource[(blockIdx.x * blockDim.x) + threadIdx.x];				\
															__syncthreads();																		\
														}
	#define FLOAT3_GLOBAL_WRITE( pdDest, shSource )		{																							\
															__syncthreads();																		\
															pdDest[(blockIdx.x * blockDim.x) + threadIdx.x] = shSource[threadIdx.x];				\
															__syncthreads();																		\
														}
#endif


static __inline__ __device__ void float3_coalescedRead( float3 * shDest, float3 const* pdSource, uint const bid, uint const bdim, uint const tid )
{
	int index = 3 * bid * bdim + tid;
	((float*)shDest)[tid] = ((float*)pdSource)[index];
	((float*)shDest)[tid+THREADSPERBLOCK] = ((float*)pdSource)[index+THREADSPERBLOCK];
	((float*)shDest)[tid+2*THREADSPERBLOCK] = ((float*)pdSource)[index+2*THREADSPERBLOCK];
}

static __inline__ __device__ void float3_CoalescedWrite( float3 * pdDest, float3 const* shSource, uint const bid, uint const bdim, uint const tid )
{
	int index = 3 * bid * bdim + tid;
	((float*)pdDest)[index] = ((float*)shSource)[tid];
	((float*)pdDest)[index+THREADSPERBLOCK] = ((float*)shSource)[tid+THREADSPERBLOCK];
	((float*)pdDest)[index+2*THREADSPERBLOCK] = ((float*)shSource)[tid+2*THREADSPERBLOCK];
}

static __inline__ __device__ __host__ int ipow( int base, int exp )
{
    int result = 1;
    while (exp)
    {
        if (exp & 1)
            result *= base;
        exp >>= 1;
        base *= base;
    }

    return result;
}

//
// Global memory
//
// VehicleData
#define STEERING(i)		pdSteering[i]
#define SPEED(i)		pdSpeed[i]
#define VELOCITY(i)		velocity( i, pdForward[i], pdSpeed[i] )
#define POSITION(i)		pdPosition[i]
#define FORWARD(i)		pdForward[i]		// TODO: remove all references to this and replace with DIRECTION.
#define DIRECTION(i)	pdDirection[i]
#define UP(i)			pdUp[i]
#define SIDE(i)			pdSide[i]

// VehicleConst
#define MAXFORCE(i)		pdMaxForce[i]
#define MASS(i)			pdMass[i]
#define MAXSPEED(i)		pdMaxSpeed[i]
#define RADIUS(i)		pdRadius[i]

//
// Shared memory
//
// VehicleData
#define STEERING_SH(i)		shSteering[i]
#define SPEED_SH(i)			shSpeed[i]
#define VELOCITY_SH(i)		velocity( i, shDirection, shSpeed )
#define POSITION_SH(i)		shPosition[i]

#define FORWARD_SH(i)		shForward[i]
#define DIRECTION_SH(i)		shDirection[i]
#define UP_SH(i)			shUp[i]
#define SIDE_SH(i)			shSide[i]

// VehicleConst
#define MAXFORCE_SH(i)		shMaxForce[i]
#define MASS_SH(i)			shMass[i]
#define MAXSPEED_SH(i)		shMaxSpeed[i]
#define RADIUS_SH(i)		shRadius[i]

#endif
