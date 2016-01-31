#include "CUDAKernelGlobals.cuh"

extern "C"
{
	__host__ void SteerToAvoidNeighborsKernelBindTextures(	float4 const*	pdBPosition,
															float4 const*	pdBDirection,
															float const*	pdBSpeed,
															float const*	pdBRadius,
															uint const		numB
															);
	__host__ void SteerToAvoidNeighborsKernelUnbindTextures( void );

	__global__ void SteerToAvoidNeighborsCUDAKernel(		uint const*		pdKNNIndices,			// In:		Indices of the KNN for each agent.
															float const*	pdKNNDistances,			// In:		Distances to the KNN for each agent.
															size_t const	k,						// In:		Number of KNN for each agent.

															// Group A data.
															float4 const*	pdPosition,				// In:		Positions of each agent.
															float4 const*	pdDirection,			// In:		Directions of facing for each agent.
															float const*	pdRadius,				// In:		Radius of each agent.
															float3 const*	pdSide,					// In:		Side direction for each agent.

															float *			pdSpeed,				// In/Out:	Speed of each agent.
															float4 *		pdSteering,				// Out:		Steering vectors for each agent.
															uint const		numA,

															// Group B data.
															uint const		numB,

															float const		minTimeToCollision,		// In:		Look-ahead time for collision avoidance.
															float const		minSeparationDistance,	// In:		Distance to consider 'close' neighbors.

															float const		fWeight,

															uint *			pdAppliedKernels,
															uint const		doNotApplyWith
															);
}

// Textures used by SteerToAvoidNeighborsCUDAKernel.
texture< float4, cudaTextureType1D, cudaReadModeElementType >	texBPosition;
texture< float4, cudaTextureType1D, cudaReadModeElementType >	texBDirection;
texture< float, cudaTextureType1D, cudaReadModeElementType >	texBSpeed;
texture< float, cudaTextureType1D, cudaReadModeElementType >	texBRadius;

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

__host__ void SteerToAvoidNeighborsKernelBindTextures(	float4 const*	pdBPosition,
														float4 const*	pdBDirection,
														float const*	pdBSpeed,
														float const*	pdBRadius,
														uint const		numB
														)
{
	static cudaChannelFormatDesc const float4ChannelDesc = cudaCreateChannelDesc< float4 >();
	static cudaChannelFormatDesc const floatChannelDesc = cudaCreateChannelDesc< float >();

	CUDA_SAFE_CALL( cudaBindTexture( NULL, texBPosition, pdBPosition, float4ChannelDesc, numB * sizeof(float4) ) );
	CUDA_SAFE_CALL( cudaBindTexture( NULL, texBDirection, pdBDirection, float4ChannelDesc, numB * sizeof(float4) ) );
	CUDA_SAFE_CALL( cudaBindTexture( NULL, texBSpeed, pdBSpeed, floatChannelDesc, numB * sizeof(float) ) );
	CUDA_SAFE_CALL( cudaBindTexture( NULL, texBRadius, pdBRadius, floatChannelDesc, numB * sizeof(float) ) );
}

__host__ void SteerToAvoidNeighborsKernelUnbindTextures( void )
{
	CUDA_SAFE_CALL( cudaUnbindTexture( texBPosition ) );
	CUDA_SAFE_CALL( cudaUnbindTexture( texBDirection ) );
	CUDA_SAFE_CALL( cudaUnbindTexture( texBSpeed ) );
	CUDA_SAFE_CALL( cudaUnbindTexture( texBRadius ) );
}

__global__ void SteerToAvoidNeighborsCUDAKernel(	uint const*		pdKNNIndices,			// In:		Indices of the KNN for each agent.
													float const*	pdKNNDistances,			// In:		Distances to the KNN for each agent.
													size_t const	k,						// In:		Number of KNN for each agent.

													// Group A data.
													float4 const*	pdPosition,				// In:		Positions of each agent.
													float4 const*	pdDirection,			// In:		Directions of facing for each agent.
													float const*	pdRadius,				// In:		Radius of each agent.
													float3 const*	pdSide,					// In:		Side direction for each agent.

													float *			pdSpeed,				// In/Out:	Speed of each agent.
													float4 *		pdSteering,				// Out:		Steering vectors for each agent.
													uint const		numA,

													// Group B data.
													uint const		numB,

													float const		minTimeToCollision,		// In:		Look-ahead time for collision avoidance.
													float const		minSeparationDistance,	// In:		Distance to consider 'close' neighbors.

													float const		fWeight,

													uint *			pdAppliedKernels,
													uint const		doNotApplyWith
												)
{
	int const index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if( index >= numA )
		return;

	if( pdAppliedKernels[ index ] & doNotApplyWith )
		return;

	float3 steering = { 0.f, 0.f, 0.f };

	extern __shared__ float shKNNDistances[];
	uint *	shKNNIndices = (uint*)shKNNDistances + (THREADSPERBLOCK*k);

	__shared__ float3 shPosition[THREADSPERBLOCK];
	__shared__ float3 shDirection[THREADSPERBLOCK];
	__shared__ float3 shSteering[THREADSPERBLOCK];
	__shared__ float3 shSide[THREADSPERBLOCK];
	__shared__ float shSpeed[THREADSPERBLOCK];
	__shared__ float shRadius[THREADSPERBLOCK];

	// Load this block's data into shared memory.
	POSITION_SH( threadIdx.x )	= POSITION_F3( threadIdx.x );
	DIRECTION_SH( threadIdx.x )	= DIRECTION_F3( threadIdx.x );
	STEERING_SH( threadIdx.x )	= STEERING_F3( threadIdx.x );
	SPEED_SH( threadIdx.x )		= SPEED( index );
	RADIUS_SH( threadIdx.x )	= RADIUS( index );
	FLOAT3_GLOBAL_READ( shSide, pdSide );
	
	// Load the KNN data from global memory.
	for( uint i = 0; i < k; i++ )
	{
		shKNNIndices[threadIdx.x*k + i] = pdKNNIndices[index*k + i];
		shKNNDistances[threadIdx.x*k + i] = pdKNNDistances[index*k + i];
	}
	__syncthreads();

	// Find the agent which is closest to collision
	float minTime = minTimeToCollision;
	float steer = 0.f;

	//uint threatIndex = UINT_MAX;				// Index of the nearest threat.
	float threatDistance = FLT_MAX;				// Distance of the nearest threat.
	float3 threatPosition;
	float3 threatDirection;
	float threatRadius;
	float threatSpeed;

	float3 threatPositionAtNearestApproach;

	// For each of the neighboring vehicles, determine which (if any)
	// pose the most immediate threat of collision.
	for( uint i = 0; i < k; i++ )
	{
		uint BIndex = shKNNIndices[threadIdx.x * k + i];

		// Check for end of KNN (will be UINT_MAX if there are no more).
		if( BIndex >= numB )
			break;

		float BDistance = shKNNDistances[threadIdx.x * k + i];

		float const		BRadius		= tex1Dfetch( texBRadius, BIndex );
		float const		BSpeed		= tex1Dfetch( texBSpeed, BIndex );
		float3 const	BPosition	= make_float3( tex1Dfetch( texBPosition, BIndex ) );
		float3 const	BDirection	= make_float3( tex1Dfetch( texBDirection, BIndex ) );

		// avoid when future positions are this close (or less)
		float const sumOfRadii = RADIUS_SH( threadIdx.x ) + BRadius;

		// Check for a 'close' neighbor.
		if( BDistance < threatDistance && BDistance < (minSeparationDistance + sumOfRadii) )
		{
			minTime = 0.f;
			//threatIndex = BIndex;
			threatDistance = BDistance;
			threatPosition = BPosition;
			threatDirection = BDirection;
			threatRadius = BRadius;
			threatSpeed = BSpeed;
			continue;
		}

		if( minTime == 0.f )
			continue;

		// predicted time until nearest approach of "this" and "other"
		float const time = predictNearestApproachTime(	POSITION_SH( threadIdx.x ), DIRECTION_SH( threadIdx.x ), SPEED_SH( threadIdx.x ),
														BPosition, BDirection, BSpeed
														);

		// If the time is in the future, sooner than any other threatened collision...
		if(	time >= 0		&&	// Time is in the future.
			time < minTime )	// Sooner than other threats.
		{
			float3 myPositionAtNearestApproach;

			// if the two will be close enough to collide, make a note of it
			if( computeNearestApproachPositions(	POSITION_SH( threadIdx.x ), DIRECTION_SH( threadIdx.x ), SPEED_SH( threadIdx.x ),
													BPosition, BDirection, BSpeed,
													time,
													threatPositionAtNearestApproach, myPositionAtNearestApproach
													)
				<
				sumOfRadii )
			{
				minTime = time;
				//threatIndex = BIndex;
				threatDistance = BDistance;
				threatPosition = BPosition;
				threatDirection = BDirection;
				threatRadius = BRadius;
				threatSpeed = BSpeed;
			}
		}
	}

	// Was there a 'threat' found?
	if( FLT_MAX != threatDistance )
	{
		float const sumOfRadii = RADIUS_SH( threadIdx.x ) + threatRadius;
		float const minCenterToCenter = minSeparationDistance + sumOfRadii;
		float3 const offset = float3_subtract( threatPosition, POSITION_SH( threadIdx.x ) );

		// parallel: +1, perpendicular: 0, anti-parallel: -1
		float const parallelness = float3_dot( DIRECTION_SH( threadIdx.x ), threatDirection );
		float const angle = 0.707f;

		if( threatDistance < minCenterToCenter )	// Other agent is within 'close' range.
		{
			// Steer hard to dodge the other agent.
			//steering = float3_perpendicularComponent( float3_minus( offset ), DIRECTION_SH( threadIdx.x ) );

			// Slow down if collision iminent
			// If the agent at threatIndex is ahead of me...
			if(	float3_dot( DIRECTION_SH( threadIdx.x ), offset ) > 0.f &&		// Other agent is in front.
				SPEED_SH( threadIdx.x ) > threatSpeed							// I am moving faster than the threat agent.
				)
			{
				// I should slow down.
				SPEED_SH( threadIdx.x ) *= (threatDistance / minCenterToCenter);
			}
		}
		else
		{
			if( parallelness < -angle )		// anti-parallel "head on" paths:
			{
				// steer away from future threat position
				float3 offset = float3_subtract( threatPositionAtNearestApproach, POSITION_SH( threadIdx.x ) );
				float sideDot = float3_dot( offset, SIDE_SH( threadIdx.x ) );
				steer = (sideDot > 0) ? -1.0f : 1.0f;
			}
			else
			{
				if (parallelness > angle)	// parallel paths: steer away from threat
				{
					float3 offset = float3_subtract( threatPosition, POSITION_SH( threadIdx.x ) );
					float sideDot = float3_dot( offset, SIDE_SH( threadIdx.x ) );
					steer = (sideDot > 0) ? -1.0f : 1.0f;
				}
				else						// perpendicular paths: steer behind threat
				{
					// (only the slower of the two does this)
					if( SPEED_SH( threadIdx.x ) <= threatSpeed )
					{
						float sideDot = float3_dot( SIDE_SH( threadIdx.x ), float3_scalar_multiply( threatDirection, threatSpeed ) );
						steer = (sideDot > 0) ? -1.0f : 1.0f;
					}
				}
			}
		steering = float3_scalar_multiply( SIDE_SH( threadIdx.x ), steer );
		}
	}

	__syncthreads();

	// Normalize and apply the weight.
	steering = float3_scalar_multiply( float3_normalize( steering ), fWeight );

	// Set the applied kernel bit.
	if( ! float3_equals( steering, float3_zero() ) )
		pdAppliedKernels[ index ] |= KERNEL_AVOID_NEIGHBORS_BIT;

	// Add into the steering vector.
	STEERING_SH( threadIdx.x ) = float3_add( steering, STEERING_SH( threadIdx.x ) );

	// Write the steering vectors and speeds to global memory.
	STEERING( index ) = STEERING_SH_F4( threadIdx.x );
	SPEED( index ) = SPEED_SH( threadIdx.x );
}
