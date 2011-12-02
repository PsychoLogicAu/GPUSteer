#include "SteerForCohesionCUDA.cuh"

#include "CUDAKernelGlobals.cuh"

#include "FlockingCommon.cuh"

extern "C"
{
	__host__ void SteerForCohesionKernelBindTextures(	float4 const*	pdBPosition,
														uint const		numB
														);
	__host__ void SteerForCohesionKernelUnindTextures( void );

	__global__ void SteerForCohesionCUDAKernel(			float4 const*	pdPosition,
														float4 const*	pdDirection,
														float4 *		pdSteering,
														size_t const	numA,

														uint const*		pdKNNIndices,
														size_t const	k,

														uint const		numB,

														float const		minDistance,
														float const		maxDistance,
														float const		cosMaxAngle,

														float const		fWeight,
														uint *			pdAppliedKernels,
														uint const		doNotApplyWith
														);
}

texture< float4, cudaTextureType1D, cudaReadModeElementType>	texBPosition;

__host__ void SteerForCohesionKernelBindTextures(	float4 const*	pdBPosition,
													uint const		numB
													)
{
	static cudaChannelFormatDesc const float4ChannelDesc = cudaCreateChannelDesc< float4 >();

	CUDA_SAFE_CALL( cudaBindTexture( NULL, texBPosition, pdBPosition, float4ChannelDesc, numB * sizeof(float4) ) );
}

__host__ void SteerForCohesionKernelUnindTextures( void )
{
	CUDA_SAFE_CALL( cudaUnbindTexture( texBPosition ) );
}

__global__ void SteerForCohesionCUDAKernel(	float4 const*	pdPosition,
											float4 const*	pdDirection,
											float4 *		pdSteering,
											size_t const	numA,

											uint const*		pdKNNIndices,
											size_t const	k,

											uint const		numB,

											float const		minDistance,
											float const		maxDistance,
											float const		cosMaxAngle,

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

	extern __shared__ uint shKNNIndices[];

	__shared__ float3 shSteering[THREADSPERBLOCK];
	__shared__ float3 shPosition[THREADSPERBLOCK];
	__shared__ float3 shDirection[THREADSPERBLOCK];

	// Copy required data from global memory.
	STEERING_SH( threadIdx.x )	= STEERING_F3( index );
	POSITION_SH( threadIdx.x )	= POSITION_F3( index );
	DIRECTION_SH( threadIdx.x )	= DIRECTION_F3( index );

	for( int i = 0; i < k; i++ )
		//shKNNIndices[threadIdx.x*k + i] = pdKNNIndices[index*k + i];
		shKNNIndices[ threadIdx.x + i * THREADSPERBLOCK ] = pdKNNIndices[ index + i * numA ];
	__syncthreads();

	    // steering accumulator and count of neighbors, both initially zero
	float3 steering = { 0.f, 0.f, 0.f };
    uint neighbors = 0;

    // For each agent in this agent's KNN neighborhood...
	for( uint i = 0; i < k; i++ )
	{
		//uint BIndex = shKNNIndices[threadIdx.x * k + i];
		uint BIndex = shKNNIndices[ threadIdx.x + i * THREADSPERBLOCK ];

		// Check for end of KNN.
		if( BIndex >= numB )
			break;

		float3 const bPosition = make_float3( tex1Dfetch( texBPosition, BIndex ) );

		if( inBoidNeighborhood( POSITION_SH( threadIdx.x ), DIRECTION_SH( threadIdx.x ), bPosition, minDistance, maxDistance, cosMaxAngle ) )
		{
			// accumulate sum of neighbor's positions
			steering = float3_add( steering, bPosition );

			// count neighbors
			neighbors++;
		}
	}

	if( neighbors > 0 )
		steering = float3_normalize( float3_subtract( float3_scalar_divide( steering, (float)neighbors ), POSITION_SH( threadIdx.x ) ) );

	// Apply the weight.
	steering = float3_scalar_multiply( steering, fWeight );

	// Set the applied kernel bit.
	if( ! float3_equals( steering, float3_zero() ) )
		pdAppliedKernels[ index ] |= KERNEL_COHESION_BIT;

	// Add into the steering vector.
	STEERING_SH( threadIdx.x ) = float3_add( steering, STEERING_SH( threadIdx.x ) );

	// Write back to global memory.
	STEERING( index ) = STEERING_SH_F4( threadIdx.x );
}
