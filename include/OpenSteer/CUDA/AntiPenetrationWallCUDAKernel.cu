#include "CUDAKernelGlobals.cuh"

extern "C"
{
	__host__ void AntiPenetrationWallKernelBindTextures(	float4 const*	pdLineStart,
															float4 const*	pdLineEnd,
															float4 const*	pdLineNormal,
															uint const		numLines
															);

	__host__ void AntiPenetrationWallKernelUnbindTextures( void );

	__global__ void AntiPenetrationWallKernel(				float4 const*	pdPosition,
															float4 *		pdDirection,
															float const*	pdSpeed,

															uint const*		pdKNLIndices,	// Indices of the K Nearest line segments...
															uint const		k,				// Number of lines in KNL.

															float const		elapsedTime,

															uint const		numAgents,
															uint const		numLines,
															uint *			pdAppliedKernels
															);
}

texture< float4, cudaTextureType1D, cudaReadModeElementType >	texLineStart;
texture< float4, cudaTextureType1D, cudaReadModeElementType >	texLineEnd;
texture< float4, cudaTextureType1D, cudaReadModeElementType >	texLineNormal;

__host__ void AntiPenetrationWallKernelBindTextures(	float4 const*	pdLineStart,
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

__host__ void AntiPenetrationWallKernelUnbindTextures( void )
{
	CUDA_SAFE_CALL( cudaUnbindTexture( texLineStart ) );
	CUDA_SAFE_CALL( cudaUnbindTexture( texLineEnd ) );
	CUDA_SAFE_CALL( cudaUnbindTexture( texLineNormal ) );
}

__global__ void AntiPenetrationWallKernel(	float4 const*	pdPosition,
											float4 *		pdDirection,
											float const*	pdSpeed,

											uint const*		pdKNLIndices,	// Indices of the K Nearest line segments...
											uint const		k,				// Number of lines in KNL.

											float const		elapsedTime,

											uint const		numAgents,
											uint const		numLines,
											uint *			pdAppliedKernels
											)
{
	int const index = (blockIdx.x * blockDim.x) + threadIdx.x;

	// Check bounds.
	if( index >= numAgents )
		return;

	// If the avoid walls kernel didn't fire for this agent... We won't need a correction.
	if( ! pdAppliedKernels[ index ] & KERNEL_AVOID_WALLS_BIT )
		return;

	// Shared memory.
	extern __shared__ uint shKNLIndices[];

	__shared__ float3	shPosition[THREADSPERBLOCK];
	__shared__ float3	shDirection[THREADSPERBLOCK];
	__shared__ float	shSpeed[THREADSPERBLOCK];

	// Copy this block's data to shared memory.
	POSITION_SH( threadIdx.x ) = POSITION_F3( index );
	DIRECTION_SH( threadIdx.x ) = DIRECTION_F3( index );
	SPEED_SH( threadIdx.x ) = SPEED( index );

	// Compute the future position of this agent.
	float3 const futurePosition = float3_add( POSITION_SH( threadIdx.x ), float3_scalar_multiply( VELOCITY_SH( threadIdx.x ), elapsedTime ) );

	float	nearestLineDistance = FLT_MAX;
	uint	nearestLineIndex = UINT_MAX;

	// For each of the K Nearest Lines...
	for( uint i = 0; i < k; i++ )
	{
		uint lineIndex = shKNLIndices[ threadIdx.x * k + i ];

		// Check for end of KNL.
		if( lineIndex >= numLines )
			break;

		float intersectDistance;
		float3 intersectPoint;
		if( LinesIntersect( POSITION_SH( threadIdx.x ), futurePosition, make_float3( tex1Dfetch( texLineStart, lineIndex) ), make_float3( tex1Dfetch( texLineEnd, lineIndex ) ), intersectPoint ) )
		{
			intersectDistance = float3_distance( POSITION_SH( threadIdx.x ), intersectPoint );

			if( intersectDistance < nearestLineDistance )
			{
				// New nearest line.
				nearestLineDistance = intersectDistance;
				nearestLineIndex = lineIndex;
			}
		}
	}

	if( UINT_MAX != nearestLineIndex )
	{
		// Non-colliding direction will be the component of direction perpendicular to the wall normal.
		DIRECTION_SH( threadIdx.x ) = float3_perpendicularComponent( DIRECTION_SH( threadIdx.x ), make_float3( tex1Dfetch( texLineNormal, nearestLineIndex ) ) );
		// Re-normalize the direction.
		DIRECTION_SH( threadIdx.x ) = float3_normalize( DIRECTION_SH( threadIdx.x ) );

		pdAppliedKernels[ index ] |= KERNEL_ANTI_PENETRATION_WALL;
	}

	DIRECTION( index ) = DIRECTION_SH_F4( threadIdx.x );
}
