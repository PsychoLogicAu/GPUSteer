//#include "KNNBruteForceCUDA.cuh"

//#include "../AgentGroupData.cuh"
#include "../VectorUtils.cuh"

#include "CUDAKernelGlobals.cuh"

// For FLT_MAX.
#include "float.h"

extern "C"
{
	// O(N2) time and memory approach.
	__global__ void KNNBruteForceCUDAKernel(	float4 const*	pdPosition,			// Agent positions.
												float *			pdDistanceMatrix,	// Global storage for distance matrix.
												size_t *		pdIndexMatrix,		// The indices which match postions in pdDistanceMatrix.
												size_t const	k,					// Number of neighbors to consider.
												size_t const	numAgents,			// Number of agents in the simulation.
												float4 const*	pdPositionOther,
												uint const		numOther,
												bool const		groupWithSelf
												);

	__global__ void KNNBruteForceCUDAKernelV2(	float4 const*	pdPosition,			// In:	Agent positions.

												uint *			pdKNNIndices,		// Out:	Indices of K Nearest Neighbors in pdPosition.
												float *			pdKNNDistances,		// Out:	Distances of each of the neighbors in pdKNNIndices.

												size_t const	k,					// In:	Number of neighbors to consider.
												size_t const	numAgents,			// In:	Number of agents in the simulation.

												float4 const*	pdPositionOther,
												uint const		numOther,
												bool const		groupWithSelf
												);

	__global__ void KNNBruteForceCUDAKernelV3(	float4 const*	pdPosition,			// Agent positions.

												uint *			pdKNNIndices,		// Output, indices of K Nearest Neighbors in pdPosition.
												float *			pdKNNDistances,		// Output, distances of the K Nearest Neighbors in pdPosition.

												size_t const	k,					// Number of neighbors to consider.
												size_t const	numAgents,			// Number of agents in the simulation.

												float4 const*	pdPositionOther,
												uint const		numOther,
												bool const		groupWithSelf,

												bool const		bSeed = false
												);

	__host__ void KNNBruteForceCUDAKernelV3BindTexture(		float4 const*	pdPositionOther,
															uint const		numOther
															);
	__host__ void KNNBruteForceCUDAKernelV3UnbindTexture( void );
}

using namespace OpenSteer;

texture< float4, cudaTextureType1D, cudaReadModeElementType >	texPositionOther;

__host__ void KNNBruteForceCUDAKernelV3BindTexture(		float4 const*	pdPositionOther,
														uint const		numOther
														)
{
	static cudaChannelFormatDesc const float4ChannelDesc = cudaCreateChannelDesc< float4 >();

	CUDA_SAFE_CALL( cudaBindTexture( NULL, texPositionOther, pdPositionOther, float4ChannelDesc, numOther * sizeof(float4) ) );
}

__host__ void KNNBruteForceCUDAKernelV3UnbindTexture( void )
{
	cudaUnbindTexture( texPositionOther );
}

__global__ void KNNBruteForceCUDAKernelV3(	float4 const*	pdPosition,			// Agent positions.

											uint *			pdKNNIndices,		// Output, indices of K Nearest Neighbors in pdPosition.
											float *			pdKNNDistances,		// Output, distances of the K Nearest Neighbors in pdPosition.

											size_t const	k,					// Number of neighbors to consider.
											size_t const	numAgents,			// Number of agents in the simulation.

											float4 const*	pdPositionOther,
											uint const		numOther,
											bool const		groupWithSelf,

											bool const		bSeed
											)
{
	int const index = (blockIdx.x * blockDim.x) + threadIdx.x;

	// Check bounds.
	if( index >= numAgents )
		return;

	// Shared memory for local priority queue computations.
	extern __shared__ uint shKNNIndices[];
	float * shKNNDistances = (float*)shKNNIndices + THREADSPERBLOCK * k;
	
	__shared__ float3 shPosition[THREADSPERBLOCK];

	POSITION_SH( threadIdx.x ) = POSITION_F3( index );

	if( bSeed )
	{
		// Load the indices of the k nearest neighbors from last frame from global memory.
		for( uint i = 0; i < k; i++ )
		{
			//shKNNIndices[ threadIdx.x * k + i ] = pdKNNIndices[ index * k + i ];
			shKNNIndices[ threadIdx.x + i * THREADSPERBLOCK ] = pdKNNIndices[ index + i * numAgents ];
			
			// Compute the current distance to this agent.
			//shKNNDistances[ threadIdx.x * k + i ] = float3_distance( POSITION_SH( threadIdx.x ), make_float3( tex1Dfetch( texPositionOther, shKNNIndices[ threadIdx.x * k + i ] ) ) );
			shKNNDistances[ threadIdx.x + i * THREADSPERBLOCK ] = float3_distance( POSITION_SH( threadIdx.x ), make_float3( tex1Dfetch( texPositionOther, shKNNIndices[ threadIdx.x + i * THREADSPERBLOCK ] ) ) );
		}

		// Re-sort using bubble sort.
		bool sorted;
		do
		{
			sorted = true;

			for( uint i = 1; i < k; i++ )
			{
				//if( shKNNDistances[ threadIdx.x * k + (i - 1) ] > shKNNDistances[ threadIdx.x * k + i ] )
				if( shKNNDistances[ threadIdx.x + (i-1) * THREADSPERBLOCK ] > shKNNDistances[ threadIdx.x + i * THREADSPERBLOCK ] )
				{
					sorted = false;

					//swap( shKNNDistances[ threadIdx.x * k + (i - 1) ], shKNNDistances[ threadIdx.x * k + i ] );
					//swap( shKNNIndices[ threadIdx.x * k + (i - 1) ], shKNNIndices[ threadIdx.x * k + i ] );
					swap( shKNNDistances[ threadIdx.x + (i-1) * THREADSPERBLOCK ], shKNNDistances[ threadIdx.x + i * THREADSPERBLOCK ] );
					swap( shKNNIndices[ threadIdx.x + (i-1) * THREADSPERBLOCK ], shKNNIndices[ threadIdx.x + i * THREADSPERBLOCK ] );
				}
			}
		} while( ! sorted );
	}
	else
	{
		// Not seeding from last frame. Set all distances to FLT_MAX.
		for( uint i = 0; i < k; i++ )
		{
			//shKNNIndices[ threadIdx.x * k + i ] = UINT_MAX;
			//shKNNDistances[ threadIdx.x * k + i ] = FLT_MAX;
			shKNNIndices[ threadIdx.x + i * THREADSPERBLOCK ] = UINT_MAX;
			shKNNDistances[ threadIdx.x + i * THREADSPERBLOCK ] = FLT_MAX;
		}
	}

	// For each of the agents...
	for( uint otherIndex = 0; otherIndex < numOther; otherIndex++ )
	{
		if( groupWithSelf && index == otherIndex )
			continue;

		bool bSeededIndex = false;
		// Make sure not adding an already seeded agent.
		for( uint i = 0; i < k; i++ )
			//if( shKNNIndices[ threadIdx.x * k + i ] == otherIndex )
			if( shKNNIndices[ threadIdx.x + i * THREADSPERBLOCK ] == otherIndex )
				bSeededIndex = true;
		if( bSeededIndex )
			continue;

		// Compute the distance between this agent and the one at i.
		float const dist = float3_distance( POSITION_SH( threadIdx.x ), make_float3( pdPositionOther[ otherIndex ] ) );

		//if( dist < shKNNDistances[(threadIdx.x * k) + (k - 1)] )	// Distance of the kth closest agent.
		if( dist < shKNNDistances[ threadIdx.x + (k-1) * THREADSPERBLOCK ] )	// Distance of the kth closest agent.
		{
			// Agent at index i is the new (at least) kth closest. Set the distance and index in shared mem.
			//shKNNDistances[(threadIdx.x * k) + (k - 1)] = dist;
			//shKNNIndices[(threadIdx.x * k) + (k - 1)] = otherIndex;
			shKNNDistances[ threadIdx.x + (k-1) * THREADSPERBLOCK ] = dist;
			shKNNIndices[ threadIdx.x + (k-1) * THREADSPERBLOCK ] = otherIndex;

			// Bubble the values up...
			for( int slot = k - 2; slot >= 0; slot-- )
			{
				//if( shKNNDistances[(threadIdx.x * k) + slot] > shKNNDistances[(threadIdx.x * k) + (slot + 1)] )
				if( shKNNDistances[ threadIdx.x + slot * THREADSPERBLOCK ] > shKNNDistances[ threadIdx.x + (slot+1) * THREADSPERBLOCK ] )
				{
					//swap( shKNNDistances[(threadIdx.x * k) + slot], shKNNDistances[(threadIdx.x * k) + (slot + 1)] );
					//swap( shKNNIndices[(threadIdx.x * k) + slot], shKNNIndices[(threadIdx.x * k) + (slot + 1)] );
					swap( shKNNDistances[ threadIdx.x + slot * THREADSPERBLOCK ], shKNNDistances[ threadIdx.x + (slot+1) * THREADSPERBLOCK ] );
					swap( shKNNIndices[ threadIdx.x + slot * THREADSPERBLOCK ], shKNNIndices[ threadIdx.x + (slot+1) * THREADSPERBLOCK ] );
				}
				else
					break;
			}
		}
	}
	
	__syncthreads();
	
	// Write the KNN indices and distances out to global memory.
	for( uint i = 0; i < k; i++ )
	{
		//pdKNNDistances[index*k + i] = shKNNDistances[threadIdx.x*k + i];
		//pdKNNIndices[index*k + i] = shKNNIndices[threadIdx.x*k + i];
		pdKNNIndices[ index + i * numAgents ] = shKNNIndices[ threadIdx.x + i * THREADSPERBLOCK ];
		pdKNNDistances[ index + i * numAgents ] = shKNNDistances[ threadIdx.x + i * THREADSPERBLOCK ];
	}
}

__global__ void KNNBruteForceCUDAKernelV2(	float4 const*	pdPosition,			// In:	Agent positions.
											uint *			pdKNNIndices,		// Out:	Indices of K Nearest Neighbors in pdPosition.
											float *			pdKNNDistances,		// Out:	Distances of each of the neighbors in pdKNNIndices.
											size_t const	k,					// In:	Number of neighbors to consider.
											size_t const	numAgents,			// In:	Number of agents in the simulation.
											float4 const*	pdPositionOther,
											uint const		numOther,
											bool const		groupWithSelf
											)
{
	int const index = (blockIdx.x * blockDim.x) + threadIdx.x;

	// Check bounds.
	if( index >= numAgents )
		return;

	// Shared memory for local computations.
	extern __shared__ uint shKNNIndices[];
	float * shKNNDistances = (float*)shKNNIndices + (THREADSPERBLOCK * k);

	__shared__ float3 shPosition[THREADSPERBLOCK];

	// Copy required data from global memory.
	POSITION_SH( threadIdx.x ) = POSITION_F3( index );

	// Set all elements of shKNNDistances to FLT_MAX and shKNNIndices to UINT_MAX.
	for( uint i = 0; i < k; i++ )
	{
		//shKNNDistances[(threadIdx.x * k) + i] = FLT_MAX;
		//shKNNIndices[(threadIdx.x * k) + i] = UINT_MAX;
		shKNNDistances[ threadIdx.x + i * THREADSPERBLOCK ] = FLT_MAX;
		shKNNIndices[ threadIdx.x + i * THREADSPERBLOCK ] = UINT_MAX;
	}
	__syncthreads();

	// TODO: This could be sped up by paging the global reads of pdPositionOther.

	// For each of the agents...
	for( uint otherIndex = 0; otherIndex < numOther; otherIndex++ )
	{
		if( groupWithSelf && index == otherIndex )
			continue;

		// Compute the distance between this agent and the one at i.
		float const dist = float3_distance( POSITION_SH( threadIdx.x ), make_float3( pdPositionOther[ otherIndex ] ) );

		if( shKNNDistances[ threadIdx.x + (k-1) * THREADSPERBLOCK ] > dist )	// Distance of the kth closest agent.
		{
			// Agent at index i is the new kth closest. Set the distance and index in shared mem.
			shKNNDistances[ threadIdx.x + (k-1) * THREADSPERBLOCK ] = dist;
			shKNNIndices[ threadIdx.x + (k-1) * THREADSPERBLOCK ] = otherIndex;

			// Bubble the values up...
			for( int slot = k - 2; slot >= 0; slot-- )
			{
				if( shKNNDistances[ threadIdx.x + slot * THREADSPERBLOCK ] > shKNNDistances[ threadIdx.x + (slot+1) * THREADSPERBLOCK ] )
				{
					swap( shKNNDistances[ threadIdx.x + slot * THREADSPERBLOCK ], shKNNDistances[ threadIdx.x + (slot+1) * THREADSPERBLOCK ] );
					swap( shKNNIndices[ threadIdx.x + slot * THREADSPERBLOCK ], shKNNIndices[ threadIdx.x + (slot+1) * THREADSPERBLOCK ] );
				}
				else
					break;
			}
		}
	}

	__syncthreads();

	// Write the KNN indices and distances out to global memory.
	for( uint i = 0; i < k; i++ )
	{
		//pdKNNDistances[index*k + i] = shKNNDistances[threadIdx.x*k + i];
		//pdKNNIndices[index*k + i] = shKNNIndices[threadIdx.x*k + i];
		pdKNNIndices[ index + i * numAgents ] = shKNNIndices[ threadIdx.x + i * THREADSPERBLOCK ];
		pdKNNDistances[ index + i * numAgents ] = shKNNDistances[ threadIdx.x + i * THREADSPERBLOCK ];
	}
}

__global__ void KNNBruteForceCUDAKernel(	float4 const*	pdPosition,			// Agent positions.
											float *			pdDistanceMatrix,	// Global storage for distance matrix.
											uint *			pdIndexMatrix,		// The indices which match postions in pdDistanceMatrix.
											uint const		k,					// Number of neighbors to consider.
											uint const		numAgents,			// Number of agents in the simulation.
											float4 const*	pdPositionOther,
											uint const		numOther,
											bool const		groupWithSelf
											)
{
	int const index = (blockIdx.x * blockDim.x) + threadIdx.x;
	int const outputIndex = index * numAgents;

	// Check bounds.
	if( index >= numAgents )
		return;

	// Copy the agent positions for this block to shared memory.
	__shared__ float3 shPosition[THREADSPERBLOCK];

	POSITION_SH( threadIdx.x ) = POSITION_F3( index );

	// TODO: This could be sped up by paging the read of pdPositionOther, and the writes of pdDistanceMatrix and pdIndexMatrix.

	// For each agent in the simulation...
	for( uint i = 0; i < numOther; i++ )
	{
		if( groupWithSelf && index == i )
		{
			pdDistanceMatrix[ outputIndex + i ] = FLT_MAX;
		}
		else
		{
			pdDistanceMatrix[ outputIndex + i ] = float3_distance( POSITION_SH( threadIdx.x ), make_float3( pdPositionOther[ i ] ) );
		}

		pdIndexMatrix[ outputIndex + i ] = i;
	}
}
