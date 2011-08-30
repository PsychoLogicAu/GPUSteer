#include "SteerForFleeCUDA.h"

#include "../VehicleGroupData.cuh"
#include "../VectorUtils.cuh"

#include "CUDAKernelGlobals.cuh"

using namespace OpenSteer;

extern "C"
{
	__global__ void SteerForFleeCUDAKernel(	float3 const* pdPosition, float3 const* pdForward, float3 * pdSteering,
											float3 const target, size_t const numAgents )
	{
		int offset = (blockIdx.x * blockDim.x) + threadIdx.x;

		// Check bounds.
		if( offset >= numAgents )
			return;

		// Shared memory for the input data.
		__shared__ float3 shSteering[THREADSPERBLOCK];
		__shared__ float3 shPosition[THREADSPERBLOCK];
		__shared__ float3 shForward[THREADSPERBLOCK];

		// Temporary shared memory storage for desired velocity.
		__shared__ float3 shDesiredVelocity[THREADSPERBLOCK];

		// Copy the required data to shared memory.
		FLOAT3_GLOBAL_READ( shSteering, pdSteering );
		FLOAT3_GLOBAL_READ( shPosition, pdPosition );
		FLOAT3_GLOBAL_READ( shForward, pdForward );

		// If we already have a steering vector set, do nothing.
		if( !float3_equals( STEERING_SH( threadIdx.x ), float3_zero() ) )
			return;

		// Get the desired velocity.
		shDesiredVelocity[ threadIdx.x ] = float3_subtract( POSITION_SH( threadIdx.x ), target );

		// Set the steering vector.
		STEERING_SH( threadIdx.x ) = float3_subtract( shDesiredVelocity[ threadIdx.x ], FORWARD_SH( threadIdx.x ) );

		__syncthreads();

		// Copy the steering vectors back to global memory.
		FLOAT3_GLOBAL_WRITE( pdSteering, shSteering );
	}
}