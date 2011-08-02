#include "SteerForPursuitCUDA.h"

#include "../VehicleGroupData.cu"
#include "../VectorUtils.cu"

#include "CUDAKernelGlobals.h"

using namespace OpenSteer;

extern "C"
{
	__global__ void SteerForPursuitCUDAKernel(	float3 * pdSteering, float3 const* pdPosition, float3 const* pdForward, float const* pdSpeed, 
												float3 const targetPosition, float3 const targetForward, float3 const targetVelocity, float const targetSpeed,
												size_t const numAgents, float const maxPredictionTime )
	{
		int offset = (blockIdx.x * blockDim.x) + threadIdx.x;

		// Check bounds.
		if( offset >= numAgents )
			return;

		// Declare shared memory.
		__shared__ float3 shSteering[THREADSPERBLOCK];
		__shared__ float3 shPosition[THREADSPERBLOCK];
		__shared__ float3 shForward[THREADSPERBLOCK];
		__shared__ float shSpeed[THREADSPERBLOCK];

		STEERING_SH( threadIdx.x ) = STEERING( offset );
		POSITION_SH( threadIdx.x ) = POSITION( offset );
		FORWARD_SH( threadIdx.x ) = FORWARD( offset );
		SPEED_SH( threadIdx.x ) = SPEED( offset );

		__syncthreads;

		// If we already have a steering vector set, do nothing.
		if( ! float3_equals( STEERING_SH( threadIdx.x ), float3_zero() ) )
			return;

		// If the target is ahead, just seek to its current position.
		float3 toTarget = float3_subtract( targetPosition, POSITION_SH( threadIdx.x ) );
		float relativeHeading = float3_dot( FORWARD_SH( threadIdx.x ), targetForward );

		if( float3_dot( toTarget, FORWARD_SH( threadIdx.x ) ) > 0 && (relativeHeading < -0.95f))
		{
			// Get the desired velocity.
			float3 desiredVelocity = float3_subtract( targetPosition, POSITION_SH( threadIdx.x ) );

			// Set the steering vector.
			STEERING_SH( threadIdx.x ) = float3_subtract( desiredVelocity, FORWARD_SH( threadIdx.x ) );
		}
		else
		{
			float lookAheadTime = float3_length( toTarget ) / ( SPEED_SH( threadIdx.x ) + targetSpeed );
			float3 newTarget = float3_add( targetPosition, float3_scalar_multiply( targetVelocity, (maxPredictionTime < lookAheadTime) ? maxPredictionTime : lookAheadTime ) );

			// Get the desired velocity.
			float3 desiredVelocity = float3_subtract( newTarget, POSITION_SH( threadIdx.x ) );

			// Set the steering vector.
			STEERING_SH( threadIdx.x ) = float3_subtract( desiredVelocity, FORWARD_SH( threadIdx.x ) );
		}

		__syncthreads;

		// Copy the steering vectors back to global memory.
		STEERING( offset ) = STEERING_SH( threadIdx.x );
	}
}