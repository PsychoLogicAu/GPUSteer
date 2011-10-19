#include "../AgentGroupData.cuh"
#include "../VectorUtils.cuh"

#include "CUDAKernelGlobals.cuh"

using namespace OpenSteer;

extern "C"
{
	
	__host__ void SteerToAvoidWallsKernelBindTextures(	float4 const*	pdLineStart,
														float4 const*	pdLineEnd,
														float4 const*	pdLineNormal,
														uint const		numLines
														);
	__host__ void SteerToAvoidWallsKernelUnbindTextures( void );

	__global__ void SteerToAvoidWallsCUDAKernel(		// Agent data.
														float4 *		pdPosition,
														float4 *		pdDirection,
														float3 const*	pdSide,
														float const*	pdSpeed,
														float const*	pdRadius,

														uint const*		pdKNLIndices,	// Indices of the K Nearest line segments...
														uint const		k,				// Number of lines in KNL.

														float const		minTimeToCollision,

														float4 *		pdSteering,

														uint const		numAgents,
														uint const		numLines,

														float const		fWeight,
														uint *			pdAppliedKernels,
														uint const		doNotApplyWith
														);
}

#define FEELER_LENGTH 4

texture< float4, cudaTextureType1D, cudaReadModeElementType >	texLineStart;
texture< float4, cudaTextureType1D, cudaReadModeElementType >	texLineEnd;
texture< float4, cudaTextureType1D, cudaReadModeElementType >	texLineNormal;

__host__ void SteerToAvoidWallsKernelBindTextures(	float4 const*	pdLineStart,
													float4 const*	pdLineEnd,
													float4 const*	pdLineNormal,
													uint const		numLines
													)
{
	static cudaChannelFormatDesc const float4ChannelDesc = cudaCreateChannelDesc< float4 >();

	CUDA_SAFE_CALL( cudaBindTexture( NULL, texLineStart, pdLineStart, float4ChannelDesc, numLines * sizeof(float4) ) );
	CUDA_SAFE_CALL( cudaBindTexture( NULL, texLineEnd, pdLineEnd, float4ChannelDesc, numLines * sizeof(float4) ) );
	CUDA_SAFE_CALL( cudaBindTexture( NULL, texLineNormal, pdLineNormal, float4ChannelDesc, numLines * sizeof(float4) ) );
}

__host__ void SteerToAvoidWallsKernelUnbindTextures( void )
{
	CUDA_SAFE_CALL( cudaUnbindTexture( texLineStart ) );
	CUDA_SAFE_CALL( cudaUnbindTexture( texLineEnd ) );
	CUDA_SAFE_CALL( cudaUnbindTexture( texLineNormal ) );
}

__global__ void SteerToAvoidWallsCUDAKernel(	// Agent data.
												float4 *		pdPosition,
												float4 *		pdDirection,
												float3 const*	pdSide,
												float const*	pdSpeed,
												float const*	pdRadius,

												uint const*		pdKNLIndices,	// Indices of the K Nearest line segments...
												uint const		k,				// Number of lines in KNL.

												float const		minTimeToCollision,

												float4 *		pdSteering,

												uint const		numAgents,
												uint const		numLines,

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

	float3 steering = { 0.f, 0.f, 0.f };
	
	// Shared memory.
	extern __shared__ uint shKNLIndices[];

	__shared__ float3 shPosition[THREADSPERBLOCK];
	__shared__ float3 shDirection[THREADSPERBLOCK];
	__shared__ float3 shSteering[THREADSPERBLOCK];
	__shared__ float3 shSide[THREADSPERBLOCK];
	__shared__ float shSpeed[THREADSPERBLOCK];
	__shared__ float shRadius[THREADSPERBLOCK];

	// Copy this block's data to shared memory.
	POSITION_SH( threadIdx.x ) = POSITION_F3( index );
	DIRECTION_SH( threadIdx.x ) = DIRECTION_F3( index );
	STEERING_SH( threadIdx.x ) = STEERING_F3( index );
	SPEED_SH( threadIdx.x ) = SPEED( index );
	RADIUS_SH( threadIdx.x ) = RADIUS( index );

	for( uint i = 0; i < k; i++ )
	{
		shKNLIndices[ threadIdx.x * k + i ] = pdKNLIndices[ index * k + i ];
	}
	FLOAT3_GLOBAL_READ( shSide, pdSide );

	// Get the agent's current velocity.
	float3 const vel = VELOCITY_SH( threadIdx.x );

	// Compute the position of the agent after lookAheadTime time, and the position of the two feelers.
	float3 feelers[] = {
		float3_add( POSITION_SH( threadIdx.x ), float3_scalar_multiply( VELOCITY_SH( threadIdx.x ), minTimeToCollision ) ),
		float3_add( 
			POSITION_SH( threadIdx.x ), 
			float3_add( 
				float3_scalar_multiply( 
					SIDE_SH( threadIdx.x ), 
					FEELER_LENGTH * RADIUS_SH( threadIdx.x )
					), 
				float3_scalar_multiply( 
					DIRECTION_SH( threadIdx.x ), 
					FEELER_LENGTH * RADIUS_SH( threadIdx.x )
					)
				)
			),
		float3_add(
			POSITION_SH( threadIdx.x ),
			float3_add(
				float3_scalar_multiply(
					float3_minus(
						SIDE_SH( threadIdx.x )
						),
					FEELER_LENGTH * RADIUS_SH( threadIdx.x )
				),
				float3_scalar_multiply(
					DIRECTION_SH( threadIdx.x ),
					FEELER_LENGTH * RADIUS_SH( threadIdx.x )
					)
				)
			)
	};

	// Index of the nearest intersecting line.
	uint	feelerIndex;
	float	nearestLineDistance = FLT_MAX;
	uint	nearestLineIndex = UINT_MAX;
	float3	nearestLineIntersectPoint;
	float3	nearestLineNormal;

	bool	touchingWall = false;

	// For each of the K Nearest Lines...
	for( uint i = 0; i < k; i++ )
	{
		uint const	lineIndex = shKNLIndices[ threadIdx.x * k + i ];

		// Check for end of KNL.
		if( lineIndex >= numLines )
			break;

		float3 intersectPoint;

		float3 const	lineNormal	= make_float3( tex1Dfetch( texLineNormal, lineIndex ) );
		float3 const	lineStart	= make_float3( tex1Dfetch( texLineStart, lineIndex) );
		float3 const	lineEnd		= make_float3( tex1Dfetch( texLineEnd, lineIndex ) );

		// Check for overlap with line.
		float3 closestPointToLine = float3_add( POSITION_SH( threadIdx.x ), float3_scalar_multiply( float3_minus( lineNormal ), RADIUS_SH( threadIdx.x ) ) );
		if( LinesIntersect( POSITION_SH( threadIdx.x ), closestPointToLine, lineStart, lineEnd, intersectPoint ) )
		{
			touchingWall = true;
			nearestLineNormal = lineNormal;
			nearestLineDistance = float3_distance( POSITION_SH( threadIdx.x ), intersectPoint );
		}

		// For each of the feelers...
		for( uint iFeeler = 0; iFeeler < 3; iFeeler++ )
		{
			float intersectDistance;
			if( LinesIntersect( POSITION_SH( threadIdx.x ), feelers[ iFeeler ], lineStart, lineEnd, intersectPoint ) )
			{
				intersectDistance = float3_distance( POSITION_SH( threadIdx.x ), intersectPoint );

				if( intersectDistance < nearestLineDistance )
				{
					// New nearest line.
					nearestLineDistance = intersectDistance;
					nearestLineIndex = lineIndex;
					nearestLineIntersectPoint = intersectPoint;
					feelerIndex = iFeeler;
					nearestLineNormal = lineNormal;
				}
			}
		}
	}

	// Enforce anti-penetration.
	if( touchingWall )
	{
		float const penetrationDistance = RADIUS_SH( threadIdx.x ) - nearestLineDistance;
		POSITION_SH( threadIdx.x ) = float3_add( POSITION_SH( threadIdx.x ), float3_scalar_multiply( nearestLineNormal, penetrationDistance ) );
		DIRECTION_SH( threadIdx.x ) = float3_normalize( float3_perpendicularComponent( DIRECTION_SH( threadIdx.x ), nearestLineNormal ) );

		// Set the wall anti-penetration bit.
		pdAppliedKernels[ index ] |= KERNEL_ANTI_PENETRATION_WALL;

	}
	else if( UINT_MAX != nearestLineIndex )
	{
		// Calculate the penetration distance.
		float const penetrationDistance = float3_length( float3_subtract( feelers[ feelerIndex ], nearestLineIntersectPoint ) );
		steering = float3_scalar_multiply( nearestLineNormal, penetrationDistance );

		// Set the applied kernel bit.
		pdAppliedKernels[ index ] |= KERNEL_AVOID_WALLS_BIT;
	}

	// Apply the weight.
	steering = float3_scalar_multiply( steering, fWeight );

	// Add into the steering vector.
	STEERING_SH( threadIdx.x ) = float3_add( steering, STEERING_SH( threadIdx.x ) );

	// Write back to global memory.
	STEERING( index ) = STEERING_SH_F4( threadIdx.x );
	POSITION( index ) = POSITION_SH_F4( threadIdx.x );
	DIRECTION( index ) = DIRECTION_SH_F4( threadIdx.x );
}
