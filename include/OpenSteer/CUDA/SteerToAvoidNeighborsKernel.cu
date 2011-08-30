#include "SteerForPursuitCUDA.h"

#include "CUDAKernelGlobals.cuh"

extern "C"
{
	__global__ void SteerToAvoidNeighborsCUDAKernel(		uint const*		pdKNNIndices,
															//float const*	pdKNNDistances,
															size_t const	k,
															
															float3 const*	pdPosition,
															float3 const*	pdDirection,
															float3 const*	pdSide,
															float const*	pdSpeed,

															float const*	pdRadius,

															float3 *		pdSteering,

															float const		minTimeToCollision,
															size_t const	numAgents
															);

	__global__ void SteerToAvoidCloseNeighborsCUDAKernel(	uint const*		pdKNNIndices,
															float const*	pdKNNDistances,
															size_t const	k,

															float3 const*	pdPosition,
															float3 const*	pdDirection,
															float const*	pdRadius,

															float3 *		pdSteering,
															float *			pdSpeed,

															float const		minSeparationDistance,

															size_t const	numAgents
															);
}

__global__ void SteerToAvoidCloseNeighborsCUDAKernel(	uint const*		pdKNNIndices,
														float const*	pdKNNDistances,
														size_t const	k,

														float3 const*	pdPosition,
														float3 const*	pdDirection,
														float const*	pdRadius,

														float3 *		pdSteering,
														float *			pdSpeed,

														float const		minSeparationDistance,

														size_t const	numAgents
														)
{
	int const index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if( index >= numAgents )
		return;

	extern __shared__ float shKNNDistances[];
	uint *	shKNNIndices = (uint*)shKNNDistances + (THREADSPERBLOCK*k);

	__shared__ float3 shPosition[THREADSPERBLOCK];
	__shared__ float3 shDirection[THREADSPERBLOCK];
	__shared__ float3 shSteering[THREADSPERBLOCK];
	__shared__ float shRadius[THREADSPERBLOCK];
	__shared__ float shSpeed[THREADSPERBLOCK];

	// Copy required global memory to shared.
	FLOAT3_GLOBAL_READ( shPosition, pdPosition );
	FLOAT3_GLOBAL_READ( shDirection, pdDirection );
	FLOAT3_GLOBAL_READ( shSteering, pdSteering );
	RADIUS_SH( threadIdx.x ) = RADIUS( index );
	__syncthreads();
	SPEED_SH( threadIdx.x ) = SPEED( index );
	__syncthreads();
	for( uint i = 0; i < k; i++ )
	{
		//shKNNIndices[threadIdx.x + (THREADSPERBLOCK*i)] = pdKNNIndices[index + (THREADSPERBLOCK*i)];
		//shKNNDistances[threadIdx.x + (THREADSPERBLOCK*i)] = pdKNNDistances[index + (THREADSPERBLOCK*i)];

		shKNNDistances[threadIdx.x*k + i] = pdKNNDistances[index*k + i];
		shKNNIndices[threadIdx.x*k + i] = pdKNNIndices[index*k + i];
	}
	__syncthreads();

	uint threatIndex;

	// For each of the KNN of this agent...
	for( int i = 0; i < k; i++ )
	{
		threatIndex = shKNNIndices[(threadIdx.x * k) + i];

		if( threatIndex >= numAgents )
			break;

		float const sumOfRadii = RADIUS_SH( threadIdx.x ) + RADIUS( threatIndex );
	    float const minCenterToCenter = minSeparationDistance + sumOfRadii;
		float3 const offset = float3_subtract( POSITION( threatIndex ), POSITION_SH( threadIdx.x ) );
		//float const currentDistance = float3_length( offset );

		// Distance was computed in KNN step. Don't waste time :)
		float const currentDistance = shKNNDistances[(threadIdx.x * k) + i];
  
		if( currentDistance < sumOfRadii )
		{
			// Agents are interpenetrating. Bad!

			// If the agent at threatIndex is ahead of me...
			if( float3_dot( DIRECTION_SH( threadIdx.x ), offset ) > 0.f )
			{
				// I should slow down.
				SPEED_SH( threadIdx.x ) *= (currentDistance / minSeparationDistance);
			}
		}
        if( currentDistance < minCenterToCenter )
        {
			// Steer hard to dodge the other agent.
			STEERING_SH( threadIdx.x ) = float3_perpendicularComponent( float3_minus( offset ), DIRECTION_SH( threadIdx.x ) );

			// TESTING: slow down if collision iminent
			// If the agent at threatIndex is ahead of me...
			//if( float3_dot( DIRECTION_SH( threadIdx.x ), offset ) > 0.f )
			//{
			//	// I should slow down.
			//	SPEED_SH( threadIdx.x ) *= (currentDistance / minCenterToCenter);
			//}
        }
	}

	__syncthreads();

	// Write the steering vectors back to global memory.
	FLOAT3_GLOBAL_WRITE( pdSteering, shSteering );

	SPEED( index ) = SPEED_SH( threadIdx.x );
}

// Given the time until nearest approach (predictNearestApproachTime)
// determine position of each agent at that time, and the distance
// between them.
__inline__ __device__ float computeNearestApproachPositions( float3 const& position, float3 const& direction, float const& speed, float3 const& otherPosition, float3 const& otherDirection, float const& otherSpeed, float const& time, float3 & threatNearestPosition, float3 & myNearestPosition )
{
	float3 const myTravel =		float3_scalar_multiply( direction, speed * time );
    float3 const otherTravel =	float3_scalar_multiply( otherDirection, otherSpeed * time );

    myNearestPosition =			float3_add( position, myTravel );
    threatNearestPosition =		float3_add( otherPosition, otherTravel );

	return float3_distance( myNearestPosition, threatNearestPosition );
}

// Given two agents, based on their current positions and velocities,
// determine the time until nearest approach.
__inline__ __device__ float predictNearestApproachTime( float3 const& position, float3 const& direction, float const& speed, float3 const& otherPosition, float3 const& otherDirection, float const& otherSpeed )
{
    // imagine we are at the origin with no velocity,
    // compute the relative velocity of the other vehicle
	float3 const myVelocity = float3_scalar_multiply( direction, speed );
	float3 otherVelocity = float3_scalar_multiply( otherDirection, otherSpeed );
    float3 const relVelocity = float3_subtract( otherVelocity, myVelocity );
	float const relSpeed = float3_length( relVelocity );

    // for parallel paths, the vehicles will always be at the same distance,
    // so return 0 (aka "now") since "there is no time like the present"
    if( relSpeed == 0.f )
		return 0.f;

    // Now consider the path of the other vehicle in this relative
    // space, a line defined by the relative position and velocity.
    // The distance from the origin (our vehicle) to that line is
    // the nearest approach.

    // Take the unit tangent along the other vehicle's path
	float3 const relTangent = float3_scalar_divide( relVelocity, relSpeed );

    // find distance from its path to origin (compute offset from
    // other to us, find length of projection onto path)
    float3 const relPosition = float3_subtract( position, otherPosition );
    float const projection = float3_dot( relTangent, relPosition );

    return projection / relSpeed;
}

__global__ void SteerToAvoidNeighborsCUDAKernel(	uint const*		pdKNNIndices,
													//float const*	pdKNNDistances,
													size_t const	k,
													
													float3 const*	pdPosition,
													float3 const*	pdDirection,
													float3 const*	pdSide,
													float const*	pdSpeed,

													float const*	pdRadius,

													float3 *		pdSteering,

													float const		minTimeToCollision,
													size_t const	numAgents
													)
{
	int const index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if( index >= numAgents )
		return;

	extern __shared__ uint shKNNIndices[];

	__shared__ float3 shPosition[THREADSPERBLOCK];
	__shared__ float3 shDirection[THREADSPERBLOCK];
	__shared__ float3 shSide[THREADSPERBLOCK];
	__shared__ float shSpeed[THREADSPERBLOCK];
	__shared__ float3 shSteering[THREADSPERBLOCK];

	// Load this block's data into shared memory.
	FLOAT3_GLOBAL_READ( shPosition, pdPosition );
	FLOAT3_GLOBAL_READ( shDirection, pdDirection );
	FLOAT3_GLOBAL_READ( shSide, pdSide );
	FLOAT3_GLOBAL_READ( shSteering, pdSteering );
	SPEED_SH( threadIdx.x ) = SPEED( index );
	__syncthreads();
	for( uint i = 0; i < k; i++ )
	{
		//shKNNIndices[threadIdx.x + (THREADSPERBLOCK*i)] = pdKNNIndices[index + (THREADSPERBLOCK*i)];
		shKNNIndices[threadIdx.x*k + i] = pdKNNIndices[index*k + i];
	}
	__syncthreads();

	if( ! float3_equals( STEERING_SH( threadIdx.x ), float3_zero() ) )
		return;

	// Find the agent which is closest to collision
	float minTime = minTimeToCollision;
	float steer = 0.f;
	uint otherIndex;
	uint threatIndex = UINT_MAX;
	float3 threatPositionAtNearestApproach;
	float3 myPositionAtNearestApproach;
	
	// For each of the neighboring vehicles, determine which (if any)
	// pose the most immediate threat of collision.
	for( uint i = 0; i < k; i++ )
	{
		otherIndex = shKNNIndices[threadIdx.x * k + i];

		// Check for end of KNN.
		if( otherIndex >= numAgents )
			break;

		// avoid when future positions are this close (or less)
		float const collisionDangerThreshold = pdRadius[index] + pdRadius[ otherIndex ];
		
		// predicted time until nearest approach of "this" and "other"
		float const time = predictNearestApproachTime(	POSITION_SH( threadIdx.x ), DIRECTION_SH( threadIdx.x ), SPEED_SH( threadIdx.x ),
														POSITION( otherIndex ), DIRECTION( otherIndex ), SPEED( otherIndex ) );
		
		// If the time is in the future, sooner than any other
		// threatened collision...
		if( time >= 0 && time < minTime )
		{
            // if the two will be close enough to collide,
            // make a note of it
            if( computeNearestApproachPositions(	POSITION_SH( threadIdx.x ), DIRECTION_SH( threadIdx.x ), SPEED_SH( threadIdx.x ),
													POSITION( otherIndex ), DIRECTION( otherIndex ), SPEED( otherIndex ),
													time,
													threatPositionAtNearestApproach, myPositionAtNearestApproach
													)
													<
													collisionDangerThreshold )
            {
                minTime = time;
                threatIndex = otherIndex;
			}
		}
	}

	if( UINT_MAX != threatIndex )
	{
        // parallel: +1, perpendicular: 0, anti-parallel: -1
		float const parallelness = float3_dot( DIRECTION_SH( threadIdx.x ), DIRECTION( threatIndex ) );
        float const angle = 0.707f;

        if( parallelness < -angle )
        {
            // anti-parallel "head on" paths:
            // steer away from future threat position
			float3 offset = float3_subtract( threatPositionAtNearestApproach, POSITION_SH( threadIdx.x ) );
			float sideDot = float3_dot( offset, SIDE_SH( threadIdx.x ) );
            steer = (sideDot > 0) ? -1.0f : 1.0f;
        }
        else
        {
            if (parallelness > angle)
            {
                // parallel paths: steer away from threat
				float3 offset = float3_subtract( POSITION( threatIndex ), POSITION_SH( threadIdx.x ) );
                float sideDot = float3_dot( offset, SIDE_SH( threadIdx.x ) );
                steer = (sideDot > 0) ? -1.0f : 1.0f;
            }
            else
            {
                // perpendicular paths: steer behind threat
                // (only the slower of the two does this)
                if( SPEED( threatIndex ) <= SPEED_SH( threadIdx.x ) )
                {
					float sideDot = float3_dot( SIDE_SH( threadIdx.x ), float3_scalar_multiply( DIRECTION( threatIndex ), SPEED( threatIndex ) ) );
                    steer = (sideDot > 0) ? -1.0f : 1.0f;
                }
            }
        }
	}

	STEERING_SH( threadIdx.x ) = float3_scalar_multiply( SIDE_SH( threadIdx.x ), steer );

	__syncthreads();

	FLOAT3_GLOBAL_WRITE( pdSteering, shSteering );
}
