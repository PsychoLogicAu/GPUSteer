#include "SteerForFleeCUDA.h"

#include "../VehicleGroupData.h"
#include "../VectorUtils.cu"

#include "CUDAKernelGlobals.h"

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
		STEERING_SH( threadIdx.x ) = STEERING( offset );
		POSITION_SH( threadIdx.x ) = POSITION( offset );
		FORWARD_SH( threadIdx.x ) = FORWARD( offset );
		
		__syncthreads();

		// If we already have a steering vector set, do nothing.
		if( !float3_equals( STEERING_SH( threadIdx.x ), float3_zero() ) )
			return;

		// Get the desired velocity.
		shDesiredVelocity[ threadIdx.x ] = float3_subtract( POSITION_SH( threadIdx.x ), target );

		// Set the steering vector.
		STEERING_SH( threadIdx.x ) = float3_subtract( shDesiredVelocity[ threadIdx.x ], FORWARD_SH( threadIdx.x ) );

		__syncthreads();

		// Copy the steering vectors back to global memory.
		STEERING( offset ) = STEERING_SH( threadIdx.x );
	}
}