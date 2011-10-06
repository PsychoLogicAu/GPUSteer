#include "SteerForEvasionCUDA.cuh"

#include "../AgentGroupData.cuh"
#include "../VectorUtils.cuh"

#include "CUDAKernelGlobals.cuh"

using namespace OpenSteer;

extern "C"
{
	__global__ void SteerForEvasionKernel(	// Agent data.
											float3 const*	pdPosition,
											float3 const*	pdDirection,
											float3 *		pdSteering,

											float3 const	menacePosition,
											float3 const	menaceDirection,
											float const		menaceSpeed,
											
											float const		maxPredictionTime,

											size_t const	numAgents,

											float const		fWeight,
											uint *			pdAppliedKernels,
											uint const		doNotApplyWith
										  );
}

__global__ void SteerForEvasionKernel(	// Agent data.
										float3 const*	pdPosition,
										float3 const*	pdDirection,
										float3 *		pdSteering,

										float3 const	menacePosition,
										float3 const	menaceDirection,
										float const		menaceSpeed,
										
										float const		maxPredictionTime,

										size_t const	numAgents,

										float const		fWeight,
										uint *			pdAppliedKernels,
										uint const		doNotApplyWith
									  )
{
	int const index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if( index >= numAgents )
		return;

	if( pdAppliedKernels[ index ] & doNotApplyWith )
		return;

	__shared__ float3 shPosition[ THREADSPERBLOCK ];
	__shared__ float3 shDirection[ THREADSPERBLOCK ];
	__shared__ float3 shSteering[ THREADSPERBLOCK ];

	FLOAT3_GLOBAL_READ( shPosition, pdPosition );
	FLOAT3_GLOBAL_READ( shDirection, pdDirection );
	FLOAT3_GLOBAL_READ( shSteering, pdSteering );

    // offset from this to menace, that distance, unit vector toward menace
    float3 const	offset			= float3_subtract( menacePosition, POSITION_SH( threadIdx.x ) );
	float const		distance		= float3_length( offset );

    float const		roughTime		= distance / menaceSpeed;
    float const		predictionTime	= ((roughTime > maxPredictionTime) ?
										maxPredictionTime :
										roughTime);

	float3 const	targetPosition	= float3_add( menacePosition, float3_scalar_multiply( float3_scalar_multiply( menaceDirection, menaceSpeed ), predictionTime ) );

	// Get the desired velocity.
	float3 const desiredVelocity = float3_subtract( POSITION_SH( threadIdx.x ), targetPosition );

	// Set the steering vector.
	float3 steering = float3_subtract( desiredVelocity, DIRECTION_SH( threadIdx.x ) );

	// Normalize and apply the weight.
	steering = float3_scalar_multiply( float3_normalize( steering ), fWeight );

	// Set the applied kernel bit.
	if( ! float3_equals( steering, float3_zero() ) )
		pdAppliedKernels[ index ] |= KERNEL_EVADE_BIT;

	// Add into the steering vector.
	STEERING_SH( threadIdx.x ) = float3_add( steering, STEERING_SH( threadIdx.x ) );

	// Copy the steering vectors back to global memory.
	FLOAT3_GLOBAL_WRITE( pdSteering, shSteering );
}