#include "SteerForSeparationCUDA.cuh"

#include "CUDAKernelGlobals.cuh"

extern "C"
{
	__global__ void SteerForSeparationKernel(	uint const*		pdKNNIndices,
												size_t const	k,
												
												float3 const*	pdPosition,
		
												float3 *		pdSteering,
												float const		weight,
												size_t const	numAgents
												);
}


__global__ void SteerForSeparationKernel(	uint const*		pdKNNIndices,
											size_t const	k,
											
											float3 const*	pdPosition,
	
											float3 *		pdSteering,
											float const		weight,
											size_t const	numAgents
											)
{
	uint index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if( index >= numAgents )
		return;

	extern __shared__ uint shKNNIndices[];

	__shared__ float3 shSteering[THREADSPERBLOCK];
	__shared__ float3 shPosition[THREADSPERBLOCK];

	// Copy required from global memory.
	FLOAT3_COALESCED_READ( shSteering, pdSteering );
	FLOAT3_COALESCED_READ( shPosition, pdPosition );
	for( uint i = 0; i < k; i++ )
		shKNNIndices[threadIdx.x + (THREADSPERBLOCK*i)] = pdKNNIndices[index + (THREADSPERBLOCK*i)];
	__syncthreads();

    // steering accumulator and count of neighbors, both initially zero
	float3 steering = make_float3( 0.f, 0.f, 0.f );
    uint neighbors = 0;
	uint otherIndex;

    // For each agent in this agent's KNN neighborhood...
	for( uint i = 0; i < k; i++ )
	{
		otherIndex = shKNNIndices[threadIdx.x * k + i];

		// Check for end of KNN.
		if( otherIndex >= numAgents )
			break;

		float3 const offset = float3_subtract( pdPosition[ otherIndex ], POSITION_SH( threadIdx.x ) );
		float const distanceSquared = float3_dot( offset, offset );
		//steering += float3_scalar_divide( offset, -distanceSquared );
		steering = float3_add( steering, float3_scalar_divide( offset, -distanceSquared ) );

		neighbors++;
	}

    // divide by neighbors, then normalize to pure direction
	if( neighbors > 0 )
		steering = float3_normalize( float3_scalar_divide( steering, (float)neighbors ) );

	// Apply the weight.
	float3_scalar_multiply( steering, weight );

	// Add into the steering vector.
	STEERING_SH( threadIdx.x ) = float3_add( steering, STEERING_SH( threadIdx.x ) );

	// Write back to global memory.
	FLOAT3_COALESCED_WRITE( pdSteering, shSteering );
}
