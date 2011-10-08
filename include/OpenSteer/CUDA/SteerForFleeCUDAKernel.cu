#include "../AgentGroupData.cuh"
#include "../VectorUtils.cuh"

#include "CUDAKernelGlobals.cuh"

using namespace OpenSteer;

extern "C"
{
	__global__ void SteerForFleeCUDAKernel(	float4 const*	pdPosition,
											float4 const*	pdDirection,
											float4 *		pdSteering,

											float3 const	target,

											size_t const	numAgents,
											float const		fWeight,
											uint *			pdAppliedKernels,
											uint const		doNotApplyWith
											);
}

__global__ void SteerForFleeCUDAKernel(	float4 const*	pdPosition,
										float4 const*	pdDirection,
										float4 *		pdSteering,

										float3 const	target,

										size_t const	numAgents,
										float const		fWeight,
										uint *			pdAppliedKernels,
										uint const		doNotApplyWith
										)

{
	int const index = (blockIdx.x * blockDim.x) + threadIdx.x;

	// Check bounds.
	if( index >= numAgents )
		return;

	if( pdAppliedKernels[ index ] & doNotApplyWith )
		return;

	// Shared memory for the input data.
	__shared__ float3 shSteering[THREADSPERBLOCK];
	__shared__ float3 shPosition[THREADSPERBLOCK];
	__shared__ float3 shDirection[THREADSPERBLOCK];

	// Copy the required data to shared memory.
	STEERING_SH( threadIdx.x ) = STEERING_F3( index );
	POSITION_SH( threadIdx.x ) = POSITION_F3( index );
	DIRECTION_SH( threadIdx.x ) = DIRECTION_F3( index );

	// Get the desired velocity.
	float3 const desiredVelocity = float3_subtract( POSITION_SH( threadIdx.x ), target );

	// Set the steering vector.
	float3 steering = float3_subtract( desiredVelocity, DIRECTION_SH( threadIdx.x ) );

	// Normalize and apply the weight.
	steering = float3_scalar_multiply( float3_normalize( steering ), fWeight );

	// Set the applied kernel bit.
	if( ! float3_equals( steering, float3_zero() ) )
		pdAppliedKernels[ index ] |= KERNEL_FLEE_BIT;

	// Add into the steering vector.
	STEERING_SH( threadIdx.x ) = float3_add( steering, STEERING_SH( threadIdx.x ) );

	// Copy the steering vectors back to global memory.
	STEERING( index ) = STEERING_SH_F4( threadIdx.x );
}
