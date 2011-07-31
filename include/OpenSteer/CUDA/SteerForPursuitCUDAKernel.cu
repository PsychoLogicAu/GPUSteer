#include "SteerForPursuitCUDA.h"

#include "../VehicleGroupData.cu"
#include "../VectorUtils.cu"

#include "CUDAKernelGlobals.h"

using namespace OpenSteer;

extern "C"
{
	__global__ void SteerForPursuitCUDAKernel(	float3 * pdSteering, float3 * pdPosition, float3 * pdForward, float * pdSpeed, 
												float3 const targetPosition, float3 const targetForward, float3 const targetVelocity, float const targetSpeed,
												size_t const numAgents, float const maxPredictionTime )
	{
		int offset = (blockIdx.x * blockDim.x) + threadIdx.x;

		// Check bounds.
		if( offset >= numAgents )
			return;

		// If we already have a steering vector set, do nothing.
		if( ! float3_equals( STEERING( offset ), float3_zero() ) )
			return;

		// If the target is ahead, just seek to its current position.
		float3 toTarget = float3_subtract( targetPosition, POSITION( offset ) );
		float relativeHeading = float3_dot( FORWARD( offset ), targetForward );

		if( float3_dot( toTarget, FORWARD( offset ) ) > 0 && (relativeHeading < -0.95f))
		{
			// Get the desired velocity.
			float3 desiredVelocity = float3_subtract( targetPosition, POSITION( offset ) );

			// Set the steering vector.
			STEERING( offset ) = float3_subtract( desiredVelocity, FORWARD( offset ) );
			return;
		}

		float lookAheadTime = float3_length( toTarget ) / (SPEED( offset ) + targetSpeed );
		float3 newTarget = float3_add( targetPosition, float3_scalar_multiply( targetVelocity, (maxPredictionTime < lookAheadTime) ? maxPredictionTime : lookAheadTime ) );

		// Get the desired velocity.
		float3 desiredVelocity = float3_subtract( newTarget, POSITION( offset ) );

		// Set the steering vector.
		STEERING( offset ) = float3_subtract( desiredVelocity, FORWARD( offset ) );
	}
}