#include "UpdateCUDA.h"

#include "../AgentGroupData.cuh"
#include "../VectorUtils.cuh"

#include "CUDAKernelGlobals.cuh"

using namespace OpenSteer;

extern "C"
{
	__global__ void UpdateCUDAKernel(	float3 *		pdSide,
										float3 *		pdUp,
										float4 *		pdDirection,
										float4 *		pdPosition,

										float4 *		pdSteering,
										float *			pdSpeed,

										float const*	pdMaxForce,
										float const*	pdMaxSpeed,
										float const*	pdMass,

										float const		elapsedTime,
										uint const		numAgents,
										uint *			pdAppliedKernels
										);
}

__global__ void UpdateCUDAKernel(	float3 *		pdSide,
									float3 *		pdUp,
									float4 *		pdDirection,
									float4 *		pdPosition,

									float4 *		pdSteering,
									float *			pdSpeed,

									float const*	pdMaxForce,
									float const*	pdMaxSpeed,
									float const*	pdMass,

									float const		elapsedTime,
									uint const		numAgents,
									uint *			pdAppliedKernels
									)
{
	int const index = (blockIdx.x * blockDim.x) + threadIdx.x;

	// Check bounds.
	if( index >= numAgents )
		return;

	// Copy the vehicleData and vehicleConst values to shared memory.
	__shared__ float3 shSide[THREADSPERBLOCK];
	__shared__ float3 shUp[THREADSPERBLOCK];
	__shared__ float3 shDirection[THREADSPERBLOCK];
	__shared__ float3 shPosition[THREADSPERBLOCK];
	__shared__ float3 shSteering[THREADSPERBLOCK];
	__shared__ float shSpeed[THREADSPERBLOCK];

	__shared__ float shMaxForce[THREADSPERBLOCK];
	__shared__ float shMaxSpeed[THREADSPERBLOCK];
	__shared__ float shMass[THREADSPERBLOCK];

	// Copy the required global memory variables to shared mem.
	DIRECTION_SH( threadIdx.x ) = DIRECTION_F3( index );
	POSITION_SH( threadIdx.x ) = POSITION_F3( index );
	STEERING_SH( threadIdx.x ) = STEERING_F3( index );
	
	SPEED_SH( threadIdx.x ) = SPEED( index );
	MAXFORCE_SH( threadIdx.x ) = MAXFORCE( index );
	MAXSPEED_SH( threadIdx.x ) = MAXSPEED( index );
	MASS_SH( threadIdx.x ) = MASS( index );

	FLOAT3_GLOBAL_READ( shSide, pdSide );
	FLOAT3_GLOBAL_READ( shUp, pdUp );

	// Set the applied kernels back to zero.
	pdAppliedKernels[ index ] = 0;

	// Enforce limit on magnitude of steering force.
	STEERING_SH( threadIdx.x ) = float3_truncateLength( STEERING_SH( threadIdx.x ), MAXFORCE_SH( threadIdx.x ) );

	// Compute acceleration and velocity.
	float3 newAcceleration = float3_scalar_divide( STEERING_SH( threadIdx.x ), MASS_SH( threadIdx.x ) );
	float3 newVelocity = float3_add( VELOCITY_SH( threadIdx.x ), float3_scalar_multiply( newAcceleration, elapsedTime ) );

	// Enforce speed limit.
	newVelocity = float3_truncateLength( newVelocity, MAXSPEED_SH( threadIdx.x ) );

	// Update speed.
	SPEED_SH( threadIdx.x ) = float3_length( newVelocity );

	if(SPEED_SH(threadIdx.x) > 0)
	{
		// Calculate the unit forward vector.
		DIRECTION_SH( threadIdx.x ) = float3_scalar_divide( newVelocity, SPEED_SH( threadIdx.x ) );

		// derive new side basis vector from NEW forward and OLD up.
		SIDE_SH( threadIdx.x ) = float3_normalize( float3_cross( DIRECTION_SH( threadIdx.x ), UP_SH( threadIdx.x ) ) );

		// derive new up basis vector from new forward and side.
		UP_SH( threadIdx.x ) = float3_cross( SIDE_SH( threadIdx.x ), DIRECTION_SH( threadIdx.x ) );
	}

	// anti-penetration with wall here

	// Euler integrate (per frame) velocity into position.
	POSITION_SH( threadIdx.x ) = float3_add( POSITION_SH( threadIdx.x ), float3_scalar_multiply( newVelocity, elapsedTime ) );

	// Set the steering vector back to zero.
	STEERING_SH( threadIdx.x ) = float3_zero();

	// Copy the shared memory back to global.
	FLOAT3_GLOBAL_WRITE( pdSide, shSide );
	FLOAT3_GLOBAL_WRITE( pdUp, shUp );

	DIRECTION( index ) = make_float4( DIRECTION_SH( threadIdx.x ), 0.f );
	POSITION( index ) = make_float4( POSITION_SH( threadIdx.x ), 0.f );
	STEERING( index ) = make_float4( STEERING_SH( threadIdx.x ), 0.f );
	SPEED( index ) = SPEED_SH( threadIdx.x );
}
