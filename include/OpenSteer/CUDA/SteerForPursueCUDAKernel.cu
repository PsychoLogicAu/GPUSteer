#include "../AgentGroupData.cuh"
#include "../VectorUtils.cuh"

#include "CUDAKernelGlobals.cuh"

using namespace OpenSteer;

extern "C"
{
__global__ void SteerForPursueCUDAKernel(	float4 const* pdPosition,
											float4 const* pdDirection,
											float const* pdSpeed, 

											float3 const targetPosition,
											float3 const targetForward,
											float3 const targetVelocity,
											float const targetSpeed,

											float4 * pdSteering,

											size_t const numAgents,
											float const maxPredictionTime,
											float const fWeight,
											uint * pdAppliedKernels,
											uint const doNotApplyWith
											);
}

__global__ void SteerForPursueCUDAKernel(	float4 const* pdPosition,
											float4 const* pdDirection,
											float const* pdSpeed, 

											float3 const targetPosition,
											float3 const targetForward,
											float3 const targetVelocity,
											float const targetSpeed,

											float4 * pdSteering,

											size_t const numAgents,
											float const maxPredictionTime,
											float const fWeight,
											uint * pdAppliedKernels,
											uint const doNotApplyWith
											)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	// Check bounds.
	if( index >= numAgents )
		return;

	if( pdAppliedKernels[ index ] & doNotApplyWith )
		return;

	// Declare shared memory.
	__shared__ float3 shPosition[THREADSPERBLOCK];
	__shared__ float3 shDirection[THREADSPERBLOCK];
	__shared__ float3 shSteering[THREADSPERBLOCK];
	__shared__ float shSpeed[THREADSPERBLOCK];

	POSITION_SH( threadIdx.x ) = POSITION_F3( index );
	DIRECTION_SH( threadIdx.x ) = DIRECTION_F3( index );
	STEERING_SH( threadIdx.x ) = STEERING_F3( index );
	SPEED_SH( threadIdx.x ) = SPEED( index );
	__syncthreads();

	float3 steering = { 0.f, 0.f, 0.f };

	// If the target is ahead, just seek to its current position.
	float3 const toTarget = float3_subtract( targetPosition, POSITION_SH( threadIdx.x ) );
	float const relativeHeading = float3_dot( DIRECTION_SH( threadIdx.x ), targetForward );

	if( (relativeHeading < -0.95f) && float3_dot( toTarget, DIRECTION_SH( threadIdx.x ) ) > 0 )
	{
		// Get the desired velocity.
		float3 const desiredVelocity = float3_subtract( targetPosition, POSITION_SH( threadIdx.x ) );

		// Set the steering vector.
		steering = float3_subtract( desiredVelocity, DIRECTION_SH( threadIdx.x ) );
	}
	else
	{
		float lookAheadTime = float3_length( toTarget ) / ( SPEED_SH( threadIdx.x ) + targetSpeed );
		float3 newTarget = float3_add( targetPosition, float3_scalar_multiply( targetVelocity, (maxPredictionTime < lookAheadTime) ? maxPredictionTime : lookAheadTime ) );

		// Get the desired velocity.
		float3 desiredVelocity = float3_subtract( newTarget, POSITION_SH( threadIdx.x ) );

		// Set the steering vector.
		steering = float3_subtract( desiredVelocity, DIRECTION_SH( threadIdx.x ) );
	}

	// Normalize and apply the weight.
	steering = float3_scalar_multiply( float3_normalize( steering ), fWeight );

	// Set the applied kernel bit.
	if( ! float3_equals( steering, float3_zero() ) )
		pdAppliedKernels[ index ] |= KERNEL_PURSUE_BIT;

	// Add into the steering vector.
	STEERING_SH( threadIdx.x ) = float3_add( steering, STEERING_SH( threadIdx.x ) );

	// Write to global memory.
	STEERING( index ) = STEERING_SH_F4( threadIdx.x );
}
