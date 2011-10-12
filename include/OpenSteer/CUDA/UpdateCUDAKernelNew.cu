#include "UpdateCUDA.h"

#include "../AgentGroupData.cuh"
#include "../VectorUtils.cuh"

#include "CUDAKernelGlobals.cuh"

using namespace OpenSteer;

extern "C"
{
	__host__ void UpdateCUDAKernelBindTextures(	float4 const*	pdLineStart,
												float4 const*	pdLineEnd,
												float4 const*	pdLineNormal,
												uint const		numLines
												);

	__host__ void UpdateCUDAKernelUnbindTextures( void );

	__global__ void UpdateCUDAKernelNew(		float3 *		pdSide,
												float3 *		pdUp,
												float4 *		pdDirection,
												float4 *		pdPosition,

												float4 *		pdSteering,
												float *			pdSpeed,

												float const*	pdMaxForce,
												float const*	pdMaxSpeed,
												float const*	pdMass,
												float const*	pdRadius,

												uint const*		pdKNLIndices,	// Indices of the K Nearest line segments...
												uint const		k,				// Number of lines in KNL.
												uint const		numLines,

												float const		elapsedTime,
												uint const		numAgents,
												uint *			pdAppliedKernels
												);
}

texture< float4, cudaTextureType1D, cudaReadModeElementType >	texLineStart;
texture< float4, cudaTextureType1D, cudaReadModeElementType >	texLineEnd;
texture< float4, cudaTextureType1D, cudaReadModeElementType >	texLineNormal;

__host__ void UpdateCUDAKernelBindTextures(	float4 const*	pdLineStart,
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

__host__ void UpdateCUDAKernelUnbindTextures( void )
{
	CUDA_SAFE_CALL( cudaUnbindTexture( texLineStart ) );
	CUDA_SAFE_CALL( cudaUnbindTexture( texLineEnd ) );
	CUDA_SAFE_CALL( cudaUnbindTexture( texLineNormal ) );
}

template < typename T >
__inline__ __host__ __device__ float clamp( T x, T a, T b)
{
	return x < a ? a : (x > b ? b : x);
}

__inline__ __host__ __device__ float3 ClosestPointOnLine( float3 const& p, float3 const& lineStart, float3 const& lineEnd )
{
	float3 const v = float3_subtract( lineEnd, lineStart );
	float3 const w = float3_subtract( p, lineStart );
	float t	= float3_dot( w, v ) / float3_dot( v, v );

	t = clamp( 0.f, 1.f, t );
	return float3_add( lineStart, float3_scalar_multiply( v, t ) );
}
	
/*
public static function closestPointLineSegment( X:Vector2D, A:Vector2D, B:Vector2D ) : Vector2D
{
    var v:Vector2D = B.minus( A );
    var w:Vector2D = X.minus( A );
    var wDotv:Number = w.dot( v );
    var t:Number = w.dot( v ) / v.dot( v );
    t = MathUtil.clamp( 0, 1, t );
    return A.plus( v.times( t ) );
}
*/

__global__ void UpdateCUDAKernelNew(	float3 *		pdSide,
										float3 *		pdUp,
										float4 *		pdDirection,
										float4 *		pdPosition,

										float4 *		pdSteering,
										float *			pdSpeed,

										float const*	pdMaxForce,
										float const*	pdMaxSpeed,
										float const*	pdMass,
										float const*	pdRadius,

										uint const*		pdKNLIndices,	// Indices of the K Nearest line segments...
										uint const		k,				// Number of lines in KNL.
										uint const		numLines,

										float const		elapsedTime,
										uint const		numAgents,
										uint *			pdAppliedKernels
										)
{
	int const index = (blockIdx.x * blockDim.x) + threadIdx.x;

	// Check bounds.
	if( index >= numAgents )
		return;

	// Shared memory.
	extern __shared__ uint shKNLIndices[];

	__shared__ float3 shSide[THREADSPERBLOCK];
	__shared__ float3 shUp[THREADSPERBLOCK];
	__shared__ float3 shDirection[THREADSPERBLOCK];
	__shared__ float3 shPosition[THREADSPERBLOCK];
	__shared__ float3 shSteering[THREADSPERBLOCK];
	__shared__ float shSpeed[THREADSPERBLOCK];

	__shared__ float shMaxForce[THREADSPERBLOCK];
	__shared__ float shMaxSpeed[THREADSPERBLOCK];
	__shared__ float shMass[THREADSPERBLOCK];
	__shared__ float shRadius[THREADSPERBLOCK];

	// Copy the required global memory variables to shared mem.
	DIRECTION_SH( threadIdx.x ) = DIRECTION_F3( index );
	POSITION_SH( threadIdx.x ) = POSITION_F3( index );
	STEERING_SH( threadIdx.x ) = STEERING_F3( index );
	
	SPEED_SH( threadIdx.x ) = SPEED( index );
	MAXFORCE_SH( threadIdx.x ) = MAXFORCE( index );
	MAXSPEED_SH( threadIdx.x ) = MAXSPEED( index );
	MASS_SH( threadIdx.x ) = MASS( index );
	RADIUS_SH( threadIdx.x ) = RADIUS( index );

	for( uint i = 0; i < k; i++ )
		shKNLIndices[ threadIdx.x * k + i ] = pdKNLIndices[ index * k + i ];

	// Set the applied kernels back to zero.
	//pdAppliedKernels[ index ] = 0;

	//FLOAT3_GLOBAL_READ( shSide, pdSide );
	FLOAT3_GLOBAL_READ( shUp, pdUp );

	// Enforce limit on magnitude of steering force.
	STEERING_SH( threadIdx.x ) = float3_truncateLength( STEERING_SH( threadIdx.x ), MAXFORCE_SH( threadIdx.x ) );

	// Compute acceleration and velocity.
	float3 newAcceleration = float3_scalar_divide( STEERING_SH( threadIdx.x ), MASS_SH( threadIdx.x ) );
	float3 newVelocity = float3_add( VELOCITY_SH( threadIdx.x ), float3_scalar_multiply( newAcceleration, elapsedTime ) );

	// Enforce speed limit.
	newVelocity = float3_truncateLength( newVelocity, MAXSPEED_SH( threadIdx.x ) );

	//
	//	Enforce anti-penetration with the walls.
	//
	// Compute the position of this agent at the end of this time step.
	float3 const futurePosition = float3_add( POSITION_SH( threadIdx.x ), float3_scalar_multiply( newVelocity, elapsedTime ) );

	// Distance and index of the nearest line.
	float	nearestLineDistance = FLT_MAX;
	uint	nearestLineIndex = UINT_MAX;
	float3	nearestLineNormal;

	// Distance and index of the nearest collision point with a line.
	float	nearestCollisionDistance = FLT_MAX;
	uint	nearestCollisionIndex = UINT_MAX;
	float3	nearestCollisionNormal;

	// For each of the K Nearest Lines...
	for( uint i = 0; i < k; i++ )
	{
		uint lineIndex = shKNLIndices[ threadIdx.x * k + i ];

		// Check for end of KNL.
		if( lineIndex >= numLines )
			break;

		// Get the normal of the line.
		float3 lineStart	= make_float3( tex1Dfetch( texLineStart, lineIndex) );
		float3 lineEnd		= make_float3( tex1Dfetch( texLineEnd, lineIndex ) );
		float3 lineNormal	= make_float3( tex1Dfetch( texLineNormal, lineIndex ) );

		float intersectDistance;
		float3 intersectPoint;
		// Check if the line intersects with the velocity during this time step.
		if( LinesIntersect( POSITION_SH( threadIdx.x ), futurePosition, lineStart, lineEnd, intersectPoint ) )
		{
			intersectDistance = float3_distance( POSITION_SH( threadIdx.x ), intersectPoint );

			if( intersectDistance < nearestCollisionDistance )
			{
				// New nearest line.
				nearestCollisionDistance = intersectDistance;
				nearestCollisionIndex = lineIndex;

				nearestCollisionNormal = lineNormal;
			}
		}

		float distanceToLineSquared = float3_distanceSquared( ClosestPointOnLine( POSITION_SH( threadIdx.x ), lineStart, lineEnd ), POSITION_SH( threadIdx.x ) );
		if( distanceToLineSquared < (RADIUS_SH( threadIdx.x ) * RADIUS_SH( threadIdx.x )) )
		{
			nearestLineDistance = sqrt( distanceToLineSquared );
			nearestLineIndex = lineIndex;
			nearestLineNormal = lineNormal;
		}
	}

	if( UINT_MAX != nearestLineIndex )
	{
			// The agent is overlapping the line. Push it away.
			POSITION_SH( threadIdx.x ) = float3_add( POSITION_SH( threadIdx.x ), float3_scalar_multiply( float3_minus( nearestLineNormal ), nearestLineDistance - RADIUS_SH( threadIdx.x ) ) );
			pdAppliedKernels[ index ] |= KERNEL_ANTI_PENETRATION_WALL;
	}

	if( UINT_MAX != nearestCollisionIndex )	// This agent's path during this time step crosses over the wall.
	{
		// Non-colliding velocity will be the component of newVelocity perpendicular to the wall normal.
		newVelocity = float3_perpendicularComponent( newVelocity, nearestCollisionNormal );	

		pdAppliedKernels[ index ] |= KERNEL_ANTI_PENETRATION_WALL;
	}

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
