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
														float4 const*	pdPosition,
														float4 const*	pdDirection,
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

__inline__ __device__ bool Intersect( float3 const& startA, float3 const& endA, float3 const& startB, float3 const& endB, float3 & intersectPoint )
{
    float const denom = ((endB.z - startB.z)*(endA.x - startA.x))	-
						((endB.x - startB.x)*(endA.z - startA.z));

    float const numera =	((endB.x - startB.x)*(startA.z - startB.z)) -
							((endB.z - startB.z)*(startA.x - startB.x));

    float const numerb =	((endA.x - startA.x)*(startA.z - startB.z)) -
							((endA.z - startA.z)*(startA.x - startB.x));

    if( fabs( denom ) < EPSILON )
    {
        if( fabs( numera ) < EPSILON && fabs( numerb ) < EPSILON )
        {
			// Lines are coincident.
			intersectPoint = startA;
            return true;
        }

		// Lines are parallel.
        return false;
    }

    float ua = numera / denom;
    float ub = numerb / denom;

    if( ua >= 0.0f && ua <= 1.0f && ub >= 0.0f && ub <= 1.0f )
    {
        // Get the intersection point.
        intersectPoint.x = startA.x + ua*(endA.x - startA.x);
		intersectPoint.y = 0.f;
        intersectPoint.z = startA.z + ua*(endA.z - startA.z);

        return true;
    }

    return false;
}

__global__ void SteerToAvoidWallsCUDAKernel(	// Agent data.
												float4 const*	pdPosition,
												float4 const*	pdDirection,
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

	// For each of the K Nearest Lines...
	for( uint i = 0; i < k; i++ )
	{
		uint lineIndex = shKNLIndices[ threadIdx.x * k + i ];

		// Check for end of KNL.
		if( lineIndex >= numLines )
			break;

		// For each of the feelers...
		for( uint iFeeler = 0; iFeeler < 3; iFeeler++ )
		{
			float intersectDistance;
			float3 intersectPoint;
			if( Intersect( POSITION_SH( threadIdx.x ), feelers[ iFeeler ], make_float3( tex1Dfetch( texLineStart, lineIndex) ), make_float3( tex1Dfetch( texLineEnd, lineIndex ) ), intersectPoint ) )
			{
				intersectDistance = float3_distance( POSITION_SH( threadIdx.x ), intersectPoint );

				if( intersectDistance < nearestLineDistance )
				{
					// New nearest line.
					nearestLineDistance = intersectDistance;
					nearestLineIndex = lineIndex;
					nearestLineIntersectPoint = intersectPoint;
					feelerIndex = iFeeler;
				}
			}
		}
	}

	if( UINT_MAX != nearestLineIndex )
	{
		// Calculate the penetration distance.
		float penetrationDistance = float3_length( float3_subtract( feelers[ feelerIndex ], nearestLineIntersectPoint ) );
		steering = float3_scalar_multiply( make_float3( tex1Dfetch( texLineNormal, nearestLineIndex ) ), penetrationDistance );
	}

	// Apply the weight.
	steering = float3_scalar_multiply( steering, fWeight );

	// Set the applied kernel bit.
	if( ! float3_equals( steering, float3_zero() ) )
		pdAppliedKernels[ index ] |= KERNEL_AVOID_WALLS_BIT;

	// Add into the steering vector.
	STEERING_SH( threadIdx.x ) = float3_add( steering, STEERING_SH( threadIdx.x ) );

	// Write back to global memory.
	STEERING( index ) = STEERING_SH_F4( threadIdx.x );
}
