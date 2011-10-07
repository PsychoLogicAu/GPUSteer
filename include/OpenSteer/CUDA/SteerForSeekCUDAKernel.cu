#include "SteerForSeekCUDA.cuh"

#include "../AgentGroupData.cuh"
#include "../VectorUtils.cuh"

#include "CUDAKernelGlobals.cuh"

using namespace OpenSteer;

extern "C"
{
	__global__ void SteerForSeekCUDAKernel(	float3 *		pdSteering,
											float3 const*	pdPosition,
											float3 const*	pdForward,
											float3 const	target,
											size_t const	numAgents,
											float const		fWeight,
											uint *			pdAppliedKernels,
											uint const		doNotApplyWith
											);
}

__global__ void SteerForSeekCUDAKernel(	float3 *		pdSteering,
										float3 const*	pdPosition,
										float3 const*	pdForward,
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

	__shared__ float3 shSteering[THREADSPERBLOCK];
	__shared__ float3 shPosition[THREADSPERBLOCK];
	__shared__ float3 shForward[THREADSPERBLOCK];

	FLOAT3_GLOBAL_READ( shSteering, pdSteering );
	FLOAT3_GLOBAL_READ( shPosition, pdPosition );
	FLOAT3_GLOBAL_READ( shForward, pdForward );

	float3 steering = { 0.f, 0.f, 0.f };

	// Get the desired velocity.
	float3 const desiredVelocity = float3_subtract( target, POSITION_SH( threadIdx.x ) );

	// Set the steering vector.
	steering = float3_subtract( desiredVelocity, FORWARD_SH( threadIdx.x ) );

	// Normalize and apply the weight.
	steering = float3_scalar_multiply( float3_normalize( steering ), fWeight );

	// Set the applied kernel bit.
	if( ! float3_equals( steering, float3_zero() ) )
		pdAppliedKernels[ index ] |= KERNEL_SEEK_BIT;

	// Add into the steering vector.
	STEERING_SH( threadIdx.x ) = float3_add( steering, STEERING_SH( threadIdx.x ) );

	// Copy the steering vectors back to global memory.
	FLOAT3_GLOBAL_WRITE( pdSteering, shSteering );
}
