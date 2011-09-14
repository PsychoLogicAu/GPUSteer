#include "UpdateCUDA.h"

#include "../VehicleGroupData.cuh"
#include "../VectorUtils.cuh"

#include "CUDAKernelGlobals.cuh"

using namespace OpenSteer;

extern "C"
{
	__global__ void UpdateCUDAKernel(	// vehicle_group_data members.
										float3 * pdSide, float3 * pdUp, float3 * pdDirection,
										float3 * pdPosition, float3 * pdSteering, float * pdSpeed,
										// vehicle_group_const members.
										float const* pdMaxForce, float const* pdMaxSpeed, float const* pdMass,
										float const elapsedTime, size_t const numAgents );
}

__global__ void UpdateCUDAKernel(		float3 * pdSide, float3 * pdUp, float3 * pdDirection,
										float3 * pdPosition, float3 * pdSteering, float * pdSpeed,
										float const* pdMaxForce, float const* pdMaxSpeed, float const* pdMass,
										float const elapsedTime, size_t const numAgents )
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

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
	FLOAT3_GLOBAL_READ( shSide, pdSide );
	FLOAT3_GLOBAL_READ( shUp, pdUp );
	FLOAT3_GLOBAL_READ( shDirection, pdDirection );
	FLOAT3_GLOBAL_READ( shPosition, pdPosition );
	FLOAT3_GLOBAL_READ( shSteering, pdSteering );
	
	SPEED_SH( threadIdx.x ) = SPEED( index );
	__syncthreads();
	MAXFORCE_SH( threadIdx.x ) = MAXFORCE( index );
	__syncthreads();
	MAXSPEED_SH( threadIdx.x ) = MAXSPEED( index );
	__syncthreads();
	MASS_SH( threadIdx.x ) = MASS( index );
	__syncthreads();

	// If we don't have a steering vector set, do nothing.
	if( float3_equals( STEERING_SH( threadIdx.x ), float3_zero() ) )
		return;

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
		// TODO: handedness? assumed right
		SIDE_SH( threadIdx.x ) = float3_normalize( float3_cross( DIRECTION_SH( threadIdx.x ), UP_SH( threadIdx.x ) ) );

		// derive new up basis vector from new forward and side.
		// TODO: handedness? assumed right
		UP_SH( threadIdx.x ) = float3_cross( SIDE_SH( threadIdx.x ), DIRECTION_SH( threadIdx.x ) );
	}

	// Euler integrate (per frame) velocity into position.
	POSITION_SH( threadIdx.x ) = float3_add( POSITION_SH( threadIdx.x ), float3_scalar_multiply( newVelocity, elapsedTime ) );

	// Set the steering vector back to zero.
	STEERING_SH( threadIdx.x ) = float3_zero();

	// Copy the shared memory back to global.
	FLOAT3_GLOBAL_WRITE( pdSide, shSide );
	FLOAT3_GLOBAL_WRITE( pdUp, shUp );
	FLOAT3_GLOBAL_WRITE( pdDirection, shDirection );
	FLOAT3_GLOBAL_WRITE( pdPosition, shPosition );
	FLOAT3_GLOBAL_WRITE( pdSteering, shSteering );

	SPEED( index ) = SPEED_SH( threadIdx.x );
}
