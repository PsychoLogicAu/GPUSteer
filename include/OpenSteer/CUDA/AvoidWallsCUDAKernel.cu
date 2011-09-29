#include "AvoidWallsCUDA.cuh"

#include "../AgentGroupData.cuh"
#include "../VectorUtils.cuh"

#include "CUDAKernelGlobals.cuh"

using namespace OpenSteer;

extern "C"
{
	__global__ void AvoidWallsCUDAKernel(	// Agent data.
											float3 const*	pdPosition,
											float3 const*	pdDirection,
											float const*	pdSpeed,

											// Wall data.
											float3 const*	pdLineStart,
											float3 const*	pdLineEnd,
											float3 const*	pdLineNormal,

											uint const*		pdKNLIndices,	// Indices of the K Nearest line segments...
											uint const		k,				// Number of lines in KNL.

											float const		minTimeToCollision,

											float3 *		pdSteering,

											size_t const	numAgents,
											float const		fWeight,

											uint *			pdAppliedKernels,
											uint const		doNotApplyWith
											);
}

//enum IntersectResult { PARALLEL, COINCIDENT, NOT_INTERESECTING, INTERESECTING };

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

__global__ void AvoidWallsCUDAKernel(	// Agent data.
										float3 const*	pdPosition,
										float3 const*	pdDirection,
										float const*	pdSpeed,

										// Wall data.
										float3 const*	pdLineStart,
										float3 const*	pdLineEnd,
										float3 const*	pdLineNormal,

										uint const*		pdKNLIndices,	// Indices of the K Nearest line segments...
										uint const		k,				// Number of lines in KNL.

										float const		minTimeToCollision,

										float3 *		pdSteering,

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

	float3 steering = { 0.f, 0.f, 0.f };
	
	// Shared memory.
	extern __shared__ uint shKNLIndices[];

	__shared__ float3 shPosition[THREADSPERBLOCK];
	__shared__ float3 shDirection[THREADSPERBLOCK];
	__shared__ float3 shSteering[THREADSPERBLOCK];
	__shared__ float shSpeed[THREADSPERBLOCK];

	// Copy this block's data to shared memory.
	FLOAT3_GLOBAL_READ( shPosition, pdPosition );
	FLOAT3_GLOBAL_READ( shDirection, pdDirection );
	FLOAT3_GLOBAL_READ( shSteering, pdSteering );
	SPEED_SH( threadIdx.x ) = SPEED( index );
	for( uint i = 0; i < k; i++ )
	{
		shKNLIndices[ threadIdx.x * k + i ] = pdKNLIndices[ index * k + i ];
	}
	__syncthreads();

	// Get the agent's current velocity.
	float3 const vel = VELOCITY_SH( threadIdx.x );

	// Compute the position of the agent after lookAheadTime time.
	float3 const futurePosition = float3_add( POSITION_SH( threadIdx.x ), float3_scalar_multiply( VELOCITY_SH( threadIdx.x ), minTimeToCollision ) );

	// Index of the nearest intersecting line.
	float nearestLineDistance = FLT_MAX;
	uint nearestLineIndex = UINT_MAX;
	float3 nearestLineIntersectPoint;

	// For each of the K Nearest Lines...
	for( uint i = 0; i < k; i++ )
	{
		uint lineIndex = shKNLIndices[ threadIdx.x * k + i ];

		// Check for end of KNL.
		if( UINT_MAX == lineIndex )
			break;

		float intersectDistance;
		float3 intersectPoint;
		if( Intersect( POSITION_SH( threadIdx.x ), futurePosition, pdLineStart[ lineIndex ], pdLineEnd[ lineIndex ], intersectPoint ) )
		{
			intersectDistance = float3_distance( POSITION_SH( threadIdx.x ), intersectPoint );

			if( intersectDistance < nearestLineDistance )
			{
				// New nearest line.
				nearestLineDistance = intersectDistance;
				nearestLineIndex = lineIndex;
				nearestLineIntersectPoint = intersectPoint;
			}
		}
	}

	if( UINT_MAX != nearestLineIndex )
	{
		// Calculate the penetration distance.
		float penetrationDistance = float3_length( float3_subtract( futurePosition, nearestLineIntersectPoint ) );
		steering = float3_scalar_multiply( pdLineNormal[ nearestLineIndex ], penetrationDistance );
	}

	__syncthreads();

	// Apply the weight.
	steering = float3_scalar_multiply( steering, fWeight );

	// Set the applied kernel bit.
	if( ! float3_equals( steering, float3_zero() ) )
		pdAppliedKernels[ index ] |= KERNEL_AVOID_WALLS_BIT;

	// Add into the steering vector.
	STEERING_SH( threadIdx.x ) = float3_add( steering, STEERING_SH( threadIdx.x ) );

	// Write back to global memory.
	FLOAT3_GLOBAL_WRITE( pdSteering, shSteering );
}