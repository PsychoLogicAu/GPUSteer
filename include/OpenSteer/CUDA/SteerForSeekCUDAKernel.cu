#include "SteerForSeekCUDA.h"

#include "../VehicleGroupData.cu"
#include "../VectorUtils.cu"

#include "CUDAKernelGlobals.h"

using namespace OpenSteer;

extern "C"
{
	__global__ void SteerForSeekCUDAKernel( float3 * pdSteering, float3 * pdPosition, float3 * pdForward, float3 target, size_t const numAgents )
	{
		int offset = (blockIdx.x * blockDim.x) + threadIdx.x;

		// Check bounds.
		if( offset >= numAgents )
			return;

		// If we already have a steering vector set, do nothing.
		if( ! float3_equals( STEERING(offset), float3_zero() ) )
			return;

		__shared__ float3 desiredVelocity[THREADSPERBLOCK];

		// Get the desired velocity.
		desiredVelocity[ threadIdx.x ] = float3_subtract( target, POSITION( offset ) );

		// Set the steering vector.
		STEERING( offset ) = float3_subtract( desiredVelocity[ threadIdx.x ], FORWARD( offset ) );
	}
}
