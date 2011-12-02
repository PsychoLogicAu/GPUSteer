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

	__global__ void SteerToAvoidNeighborsCUDAKernel(		// KNN data.
															uint const*		pdKNNIndices,
															size_t const	k,
															
															// Group A data.
															float4 const*	pdPosition,
															float4 const*	pdDirection,
															float3 const*	pdSide,
															float const*	pdRadius,

															float const*	pdSpeed,
															float4 *		pdSteering,
															size_t const	numA,

															// Group B data.
															uint const		numB,

															float const		minTimeToCollision,

															float const		fWeight,

															uint *			pdAppliedKernels,
															uint const		doNotApplyWith
															);

	__global__ void SteerToAvoidCloseNeighborsCUDAKernel(	// KNN data.
															uint const*		pdKNNIndices,
															float const*	pdKNNDistances,
															size_t const	k,

															// Group A data.
															float4 const*	pdPosition,
															float4 const*	pdDirection,
															float const*	pdRadius,

															float4 *		pdSteering,
															size_t const	numA,

															// Group B data.
															uint const		numB,

															float const		minSeparationDistance,

															float const		fWeight,

															uint *			pdAppliedKernels,
															uint const		doNotApplyWith
															);
}

// Textures used by SteerToAvoidNeighborsCUDAKernel & SteerToAvoidCloseNeighborsCUDAKernel.
texture< float4, cudaTextureType1D, cudaReadModeElementType >	texBPosition;
texture< float4, cudaTextureType1D, cudaReadModeElementType >	texBDirection;
texture< float, cudaTextureType1D, cudaReadModeElementType >	texBSpeed;
texture< float, cudaTextureType1D, cudaReadModeElementType >	texBRadius;

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

__global__ void SteerToAvoidCloseNeighborsCUDAKernel(	// KNN data.
														uint const*		pdKNNIndices,
														float const*	pdKNNDistances,
														size_t const	k,

														// Group A data.
														float4 const*	pdPosition,
														float4 const*	pdDirection,
														float const*	pdRadius,

														float4 *		pdSteering,
														size_t const	numA,

														// Group B data.
														uint const		numB,

														float const		minSeparationDistance,

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
	__shared__ float shRadius[THREADSPERBLOCK];

	// Copy required global memory to shared.
	POSITION_SH( threadIdx.x ) = POSITION_F3( index );
	DIRECTION_SH( threadIdx.x ) = DIRECTION_F3( index );
	STEERING_SH( threadIdx.x ) = STEERING_F3( index );
	RADIUS_SH( threadIdx.x ) = RADIUS( index );
	for( uint i = 0; i < k; i++ )
	{
		//shKNNDistances[threadIdx.x*k + i] = pdKNNDistances[index*k + i];
		//shKNNIndices[threadIdx.x*k + i] = pdKNNIndices[index*k + i];
		shKNNIndices[ threadIdx.x + i * THREADSPERBLOCK ] = pdKNNIndices[ index + i * numA ];
		shKNNDistances[ threadIdx.x + i * THREADSPERBLOCK ] = pdKNNDistances[ index + i * numA ];
	}

	// For each of the KNN of this agent...
	for( int i = 0; i < k; i++ )
	{
		//uint const threatIndex = shKNNIndices[(threadIdx.x * k) + i];
		uint const threatIndex = shKNNIndices[ threadIdx.x + i * THREADSPERBLOCK ];

		// Check for the end of KNN.
		if( threatIndex >= numB )
			break;

		// Get the distance to the other agent.
		//float const threatDistance = shKNNDistances[(threadIdx.x * k) + i];
		float const threatDistance = shKNNDistances[ threadIdx.x + i * THREADSPERBLOCK ];

		float const sumOfRadii = RADIUS_SH( threadIdx.x ) + tex1Dfetch( texBRadius, threatIndex );
	    float const minCenterToCenter = minSeparationDistance + sumOfRadii;
  
        if( threatDistance < minCenterToCenter )
        {
			float3 const offset = float3_subtract( make_float3( tex1Dfetch( texBPosition, threatIndex ) ), POSITION_SH( threadIdx.x ) );

			// Steer hard to dodge the other agent.
			steering = float3_perpendicularComponent( float3_minus( offset ), DIRECTION_SH( threadIdx.x ) );
			break;
        }
	}

	// Normalize and apply the weight.
	steering = float3_scalar_multiply( float3_normalize( steering ), fWeight );

	// Set the applied kernel bit.
	if( ! float3_equals( steering, float3_zero() ) )
		pdAppliedKernels[ index ] |= KERNEL_AVOID_CLOSE_NEIGHBORS_BIT;

	// Add into the steering vector.
	STEERING_SH( threadIdx.x ) = float3_add( steering, STEERING_SH( threadIdx.x ) );

	// Write the steering vectors back to global memory.
	STEERING( index ) = STEERING_SH_F4( threadIdx.x );
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

__global__ void SteerToAvoidNeighborsCUDAKernel(		// KNN data.
														uint const*		pdKNNIndices,
														size_t const	k,
														
														// Group A data.
														float4 const*	pdPosition,
														float4 const*	pdDirection,
														float3 const*	pdSide,
														float const*	pdRadius,

														float const*	pdSpeed,
														float4 *		pdSteering,
														size_t const	numA,

														// Group B data.
														uint const		numB,

														float const		minTimeToCollision,

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

	extern __shared__ uint shKNNIndices[];

	__shared__ float3 shPosition[THREADSPERBLOCK];
	__shared__ float3 shDirection[THREADSPERBLOCK];
	__shared__ float3 shSide[THREADSPERBLOCK];
	__shared__ float3 shSteering[THREADSPERBLOCK];
	__shared__ float shSpeed[THREADSPERBLOCK];
	__shared__ float shRadius[THREADSPERBLOCK];

	// Load this block's data into shared memory.
	for( uint i = 0; i < k; i++ )
	{
		//shKNNIndices[threadIdx.x*k + i] = pdKNNIndices[index*k + i];
		shKNNIndices[ threadIdx.x + i * THREADSPERBLOCK ] = pdKNNIndices[ index + i * numA ];
	}

	POSITION_SH( threadIdx.x ) = POSITION_F3( index );
	DIRECTION_SH( threadIdx.x ) = DIRECTION_F3( index );
	STEERING_SH( threadIdx.x ) = STEERING_F3( index );
	SPEED_SH( threadIdx.x ) = SPEED( index );
	RADIUS_SH( threadIdx.x ) = RADIUS( index );
	FLOAT3_GLOBAL_READ( shSide, pdSide );

	// Find the agent which is closest to collision
	float minTime = minTimeToCollision;
	float steer = 0.f;
	float3 threatPositionAtNearestApproach;
	float3 myPositionAtNearestApproach;

	uint threatIndex = UINT_MAX;
	float3	threatDirection;
	float3	threatPosition;
	float	threatSpeed;
	
	// For each of the neighboring vehicles, determine which (if any)
	// pose the most immediate threat of collision.
	for( uint i = 0; i < k; i++ )
	{
		//uint const BIndex = shKNNIndices[threadIdx.x * k + i];
		uint const BIndex = shKNNIndices[ threadIdx.x + i * THREADSPERBLOCK ];

		// Check for end of KNN.
		if( BIndex >= numB )
			break;

		float const		BRadius		= tex1Dfetch( texBRadius, BIndex );
		float const		BSpeed		= tex1Dfetch( texBSpeed, BIndex );
		float3 const	BPosition	= make_float3( tex1Dfetch( texBPosition, BIndex ) );
		float3 const	BDirection	= make_float3( tex1Dfetch( texBDirection, BIndex ) );


		// avoid when future positions are this close (or less)
		float const collisionDangerThreshold = RADIUS_SH( threadIdx.x ) + BRadius;
		
		// predicted time until nearest approach of "this" and "other"
		float const time = predictNearestApproachTime(	POSITION_SH( threadIdx.x ), DIRECTION_SH( threadIdx.x ), SPEED_SH( threadIdx.x ),
														BPosition, BDirection, BSpeed );
		
		// If the time is in the future, sooner than any other
		// threatened collision...
		if( time >= 0 && time < minTime )
		{
            // if the two will be close enough to collide,
            // make a note of it
            if( computeNearestApproachPositions(	POSITION_SH( threadIdx.x ), DIRECTION_SH( threadIdx.x ), SPEED_SH( threadIdx.x ),
													BPosition, BDirection, BSpeed,
													time,
													threatPositionAtNearestApproach, myPositionAtNearestApproach
													)
													<
													collisionDangerThreshold )
            {
                minTime			= time;
                threatIndex		= BIndex;
				threatDirection	= BDirection;
				threatSpeed		= BSpeed;
			}
		}
	}

	// If there was no threat found, return.
	if( UINT_MAX == threatIndex )
		return;


    // parallel: +1, perpendicular: 0, anti-parallel: -1
	float const parallelness = float3_dot( DIRECTION_SH( threadIdx.x ), threatDirection );
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
        if( parallelness > angle )
        {
            // parallel paths: steer away from threat
			float3 offset = float3_subtract( threatPosition, POSITION_SH( threadIdx.x ) );
            float sideDot = float3_dot( offset, SIDE_SH( threadIdx.x ) );
            steer = (sideDot > 0) ? -1.0f : 1.0f;
        }
        else
        {
            // perpendicular paths: steer behind threat
            // (only the slower of the two does this)
            if( SPEED_SH( threadIdx.x ) <= threatSpeed )
            {
				float sideDot = float3_dot( SIDE_SH( threadIdx.x ), float3_scalar_multiply( threatDirection, threatSpeed ) );
                steer = (sideDot > 0) ? -1.0f : 1.0f;
            }
        }
    }

	steering = float3_scalar_multiply( SIDE_SH( threadIdx.x ), steer );

	// Normalize and apply the weight.
	steering = float3_scalar_multiply( float3_normalize( steering ), fWeight );

	// Set the applied kernel bit.
	if( ! float3_equals( steering, float3_zero() ) )
		pdAppliedKernels[ index ] |= KERNEL_AVOID_NEIGHBORS_BIT;

	// Add into the steering vector.
	STEERING_SH( threadIdx.x ) = float3_add( steering, STEERING_SH( threadIdx.x ) );

	// Write the steering vectors to global memory.
	STEERING( index ) = STEERING_SH_F4( threadIdx.x );
}
