#include "CUDAKernelGlobals.cuh"

extern "C"
{
	__host__ void AntiPenetrationAgentsKernelBindTextures(	float4 const*	pdBPosition,
															float const*	pdBRadius,
															uint const*		pdBAppliedKernels,
															uint const		numB
															);

	__host__ void AntiPenetrationAgentsKernelUnbindTextures( void );

	__global__ void AntiPenetrationAgentsCUDAKernel(		float4 const*	pdPosition,
															float const*	pdRadius,
															uint const		numA,

															uint const*		pdKNNIndices,
															float const*	pdKNNDistances,
															uint const		k,

															uint const		numB,

															float4 *		pdPositionOut,
															
															uint *			pdAppliedKernels,
															uint const		doNotApplyWith
															);
}

texture< float4, cudaTextureType1D, cudaReadModeElementType >	texBPosition;
texture< float, cudaTextureType1D, cudaReadModeElementType >	texBRadius;
texture< uint, cudaTextureType1D, cudaReadModeElementType >		texBAppliedKernels;

__host__ void AntiPenetrationAgentsKernelBindTextures(	float4 const*	pdBPosition,
														float const*	pdBRadius,
														uint const*		pdBAppliedKernels,
														uint const		numB
														)
{
	static cudaChannelFormatDesc const float4ChannelDesc = cudaCreateChannelDesc< float4 >();
	static cudaChannelFormatDesc const floatChannelDesc = cudaCreateChannelDesc< float >();
	static cudaChannelFormatDesc const uintChannelDesc = cudaCreateChannelDesc< uint >();

	CUDA_SAFE_CALL( cudaBindTexture( NULL, texBPosition, pdBPosition, float4ChannelDesc, numB * sizeof(float4) ) );
	CUDA_SAFE_CALL( cudaBindTexture( NULL, texBRadius, pdBRadius, floatChannelDesc, numB * sizeof(float) ) );
	CUDA_SAFE_CALL( cudaBindTexture( NULL, texBAppliedKernels, pdBAppliedKernels, uintChannelDesc, numB * sizeof(uint) ) );
}

__host__ void AntiPenetrationAgentsKernelUnbindTextures( void )
{
	CUDA_SAFE_CALL( cudaUnbindTexture( texBPosition ) );
	CUDA_SAFE_CALL( cudaUnbindTexture( texBRadius ) );
	CUDA_SAFE_CALL( cudaUnbindTexture( texBAppliedKernels ) );
}


__global__ void AntiPenetrationAgentsCUDAKernel(	float4 const*	pdPosition,
													float const*	pdRadius,
													uint const		numA,

													uint const*		pdKNNIndices,
													float const*	pdKNNDistances,
													uint const		k,

													uint const		numB,

													float4 *		pdPositionOut,
													
													uint *			pdAppliedKernels,
													uint const		doNotApplyWith
													)
{
	int const index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if( index >= numA )
		return;

	if( pdAppliedKernels[ index ] & doNotApplyWith )
		return;

	__shared__ uint		shKNNIndices[THREADSPERBLOCK];
	__shared__ float	shKNNDistances[THREADSPERBLOCK];
	__shared__ float3	shPosition[THREADSPERBLOCK];
	__shared__ float	shRadius[THREADSPERBLOCK];

	POSITION_SH( threadIdx.x ) = POSITION_F3( index );
	RADIUS_SH( threadIdx.x ) = RADIUS( index );

	// Load the data for the first KNN for each agent.
	shKNNIndices[ threadIdx.x ] = pdKNNIndices[ index ];
	shKNNDistances[ threadIdx.x ] = pdKNNDistances[ index ];

	if( shKNNIndices[ threadIdx.x ] >= numB )	// There is no near agent.
		return;

	float const		sumOfRadii = RADIUS_SH( threadIdx.x ) + tex1Dfetch( texBRadius, shKNNIndices[ threadIdx.x ] );
	float const		overlapDist = sumOfRadii - shKNNDistances[ threadIdx.x ];

	if( overlapDist < 0 )	// Agents did not overlap.
		return;

	float3 const	offset = float3_subtract( POSITION_SH( threadIdx.x ), make_float3( tex1Dfetch( texBPosition, shKNNIndices[ threadIdx.x ] ) ) );
	float3 const	offsetNorm = float3_normalize( offset );

	float const		moveDist = tex1Dfetch( texBAppliedKernels, shKNNIndices[ threadIdx.x ] ) & doNotApplyWith ? overlapDist : 0.5f * overlapDist;

	pdPositionOut[ index ] = make_float4( float3_add( POSITION_SH( threadIdx.x ), float3_scalar_multiply( offsetNorm, moveDist ) ), 0.f );
	pdAppliedKernels[ index ] |= KERNEL_ANTI_PENETRATION_AGENT;
}
