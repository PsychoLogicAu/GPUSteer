#include "KNNBruteForceCUDA.cuh"

//#include "../VehicleGroupData.cuh"
#include "../VectorUtils.cuh"

#include "CUDAKernelGlobals.cuh"

// For FLT_MAX.
#include "float.h"

using namespace OpenSteer;

extern "C"
{
	// O(N2) time and memory approach.
	__global__ void KNNBruteForceCUDAKernel(	float3 const*	pdPosition,			// Agent positions.
												float *			pdDistanceMatrix,	// Global storage for distance matrix.
												size_t *		pdIndexMatrix,		// The indices which match postions in pdDistanceMatrix.
												size_t const	k,					// Number of neighbors to consider.
												size_t const	numAgents,			// Number of agents in the simulation.
												float3 const*	pdPositionOther,
												uint const		numOther,
												bool const		groupWithSelf
												);

	__global__ void KNNBruteForceCUDAKernelV2(	float3 const*	pdPosition,			// In:	Agent positions.

												uint *			pdKNNIndices,		// Out:	Indices of K Nearest Neighbors in pdPosition.
												float *			pdKNNDistances,		// Out:	Distances of each of the neighbors in pdKNNIndices.

												size_t const	k,					// In:	Number of neighbors to consider.
												size_t const	numAgents,			// In:	Number of agents in the simulation.

												float3 const*	pdPositionOther,
												uint const		numOther,
												bool const		groupWithSelf
												);

	__global__ void KNNBruteForceCUDAKernelV3(	float3 const*	pdPosition,			// Agent positions.

												uint *			pdKNNIndices,		// Output, indices of K Nearest Neighbors in pdPosition.
												float *			pdKNNDistances,		// Output, distances of the K Nearest Neighbors in pdPosition.

												size_t const	k,					// Number of neighbors to consider.
												size_t const	numAgents,			// Number of agents in the simulation.

												float3 const*	pdPositionOther,
												uint const		numOther,
												bool const		groupWithSelf,

												bool const		bSeed = false
												);
	
}

__global__ void KNNBruteForceCUDAKernelV3(	float3 const*	pdPosition,			// Agent positions.

											uint *			pdKNNIndices,		// Output, indices of K Nearest Neighbors in pdPosition.
											float *			pdKNNDistances,		// Output, distances of the K Nearest Neighbors in pdPosition.

											size_t const	k,					// Number of neighbors to consider.
											size_t const	numAgents,			// Number of agents in the simulation.

											float3 const*	pdPositionOther,
											uint const		numOther,
											bool const		groupWithSelf,

											bool const		bSeed = false
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

	FLOAT3_GLOBAL_READ( shPosition, pdPosition );
	
	// Set all elements of shKNNDistances to FLT_MAX.
	for( uint i = 0; i < k; i++ )
	{
		shKNNIndices[(threadIdx.x * k) + i] = UINT_MAX;
		shKNNDistances[(threadIdx.x * k) + i] = FLT_MAX;
	}

	__syncthreads();
	
	if( bSeed )
	{
		// Set the seeding values from the previous update.
		for( uint i = 0; i < k; i++ )
		{
			// Get the index of the ith closest agent from the last frame.
			uint const ind = pdKNNIndices[index * k + i];

			if( UINT_MAX == ind )
				continue;

			// Compute the distance between this agent and the one at index.
			float const dist = float3_distance( POSITION_SH( threadIdx.x ), pdPositionOther[ ind ] );

			if( dist < shKNNDistances[(threadIdx.x * k) + (k - 1)] )	// Distance of the kth closest agent.
			{
				// Agent at index i is the new kth closest. Set the distance and index in shared mem.
				shKNNDistances[(threadIdx.x * k) + (k - 1)] = dist;
				shKNNIndices[(threadIdx.x * k) + (k - 1)] = ind;

				// Bubble the values up... this is necessary as their positions may have changed relative to each other since the last update.
				for( int slot = k - 2; slot >= 0; slot-- )
				{
					if( shKNNDistances[(threadIdx.x * k) + slot] > shKNNDistances[(threadIdx.x * k) + (slot + 1)] )
					{
						swap( shKNNDistances[(threadIdx.x * k) + slot], shKNNDistances[(threadIdx.x * k) + (slot + 1)] );
						swap( shKNNIndices[(threadIdx.x * k) + slot], shKNNIndices[(threadIdx.x * k) + (slot + 1)] );
					}
					else
						break;
				}
			}
		}
	}

	__syncthreads();

	// For each of the agents...
	for( uint otherIndex = 0; otherIndex < numOther; otherIndex++ )
	{
		if( groupWithSelf && index == otherIndex )
			continue;

		bool bSeededIndex = false;
		// Make sure not adding an already seeded agent.
		for( uint i = 0; i < k; i++ )
			if( shKNNIndices[ threadIdx.x * k + i ] == otherIndex )
				bSeededIndex = true;
		if( bSeededIndex )
			continue;

		// Compute the distance between this agent and the one at i.
		float const dist = float3_distance( position, pdPositionOther[ otherIndex ] );

		if( dist < shKNNDistances[(threadIdx.x * k) + (k - 1)] )	// Distance of the kth closest agent.
		{
			// Agent at index i is the new (at least) kth closest. Set the distance and index in shared mem.
			shKNNDistances[(threadIdx.x * k) + (k - 1)] = dist;
			shKNNIndices[(threadIdx.x * k) + (k - 1)] = otherIndex;

			// Bubble the values up...
			for( int slot = k - 2; slot >= 0; slot-- )
			{
				if( shKNNDistances[(threadIdx.x * k) + slot] > shKNNDistances[(threadIdx.x * k) + (slot + 1)] )
				{
					swap( shKNNDistances[(threadIdx.x * k) + slot], shKNNDistances[(threadIdx.x * k) + (slot + 1)] );
					swap( shKNNIndices[(threadIdx.x * k) + slot], shKNNIndices[(threadIdx.x * k) + (slot + 1)] );
				}
				else
					break;
			}
		}
	}
	
	__syncthreads();
	
	// Write the KNN indices and distances out to global memory.
	if( blockIdx.x < gridDim.x-1 )
	{
		for( uint i = 0; i < k; i++ )
		{
			pdKNNDistances[index*k + THREADSPERBLOCK*i] = shKNNDistances[threadIdx.x*k + THREADSPERBLOCK*i];
			pdKNNIndices[index*k + THREADSPERBLOCK*i] = shKNNIndices[threadIdx.x*k + THREADSPERBLOCK*i];
		}
	}
	else
	{
		for( uint i = 0; i < k; i++ )
		{
			pdKNNDistances[index*k + i] = shKNNDistances[threadIdx.x*k + i];
			pdKNNIndices[index*k + i] = shKNNIndices[threadIdx.x*k + i];
		}
	}
}

__global__ void KNNBruteForceCUDAKernelV2(	float3 const*	pdPosition,			// In:	Agent positions.
											uint *			pdKNNIndices,		// Out:	Indices of K Nearest Neighbors in pdPosition.
											float *			pdKNNDistances,		// Out:	Distances of each of the neighbors in pdKNNIndices.
											size_t const	k,					// In:	Number of neighbors to consider.
											size_t const	numAgents,			// In:	Number of agents in the simulation.
											float3 const*	pdPositionOther,
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
	FLOAT3_GLOBAL_READ( shPosition, pdPosition );

	// Set all elements of shKNNDistances to FLT_MAX and shKNNIndices to UINT_MAX.
	for( uint i = 0; i < k; i++ )
	{
		shKNNDistances[(threadIdx.x * k) + i] = FLT_MAX;
		shKNNIndices[(threadIdx.x * k) + i] = UINT_MAX;
	}
	__syncthreads();

	// TODO: This could be sped up by paging the global reads of pdPositionOther.

	// For each of the agents...
	for( uint otherIndex = 0; otherIndex < numOther; otherIndex++ )
	{
		if( groupWithSelf && index == otherIndex )
			continue;

		// Compute the distance between this agent and the one at i.
		float const dist = float3_distance( POSITION_SH( threadIdx.x ), pdPositionOther[ otherIndex ] );

		if( shKNNDistances[(threadIdx.x * k) + (k - 1)] > dist )	// Distance of the kth closest agent.
		{
			// Agent at index i is the new kth closest. Set the distance and index in shared mem.
			shKNNDistances[(threadIdx.x * k) + (k - 1)] = dist;
			shKNNIndices[(threadIdx.x * k) + (k - 1)] = otherIndex;

			// Bubble the values up...
			for( int slot = k - 2; slot >= 0; slot-- )
			{
				if( shKNNDistances[(threadIdx.x * k) + slot] > shKNNDistances[(threadIdx.x * k) + (slot + 1)] )
				{
					swap( shKNNDistances[(threadIdx.x * k) + slot], shKNNDistances[(threadIdx.x * k) + (slot + 1)] );
					swap( shKNNIndices[(threadIdx.x * k) + slot], shKNNIndices[(threadIdx.x * k) + (slot + 1)] );
				}
				else
					break;
			}
		}
	}

	__syncthreads();

	// Write the KNN indices and distances out to global memory.
	if( blockIdx.x < gridDim.x-1 )
	{
		for( uint i = 0; i < k; i++ )
		{
			pdKNNDistances[index*k + THREADSPERBLOCK*i] = shKNNDistances[threadIdx.x*k + THREADSPERBLOCK*i];
			pdKNNIndices[index*k + THREADSPERBLOCK*i] = shKNNIndices[threadIdx.x*k + THREADSPERBLOCK*i];
		}
	}
	else
	{
		for( uint i = 0; i < k; i++ )
		{
			pdKNNDistances[index*k + i] = shKNNDistances[threadIdx.x*k + i];
			pdKNNIndices[index*k + i] = shKNNIndices[threadIdx.x*k + i];
		}
	}
}

__global__ void KNNBruteForceCUDAKernel(	float3 const*	pdPosition,			// Agent positions.
											float *			pdDistanceMatrix,	// Global storage for distance matrix.
											size_t *		pdIndexMatrix,		// The indices which match postions in pdDistanceMatrix.
											//size_t *		pdKNNIndices,		// Output, indices of K Nearest Neighbors in pdPosition.
											size_t const	k,					// Number of neighbors to consider.
											size_t const	numAgents			// Number of agents in the simulation.
										)
{
	int offset = (blockIdx.x * blockDim.x) + threadIdx.x;
	int outputOffset = offset * numAgents;

	// Check bounds.
	if( offset >= numAgents )
		return;

	// Copy the agent positions for this block to shared memory.
	__shared__ float3 shPosition[THREADSPERBLOCK];
	POSITION_SH( threadIdx.x ) = POSITION( offset );

	// For each agent in the simulation...
	for( size_t i = 0; i < numAgents; i++ )
	{
		pdDistanceMatrix[ outputOffset + i ] = float3_distance( POSITION_SH( threadIdx.x ), pdPosition[ i ] );
		pdIndexMatrix[ outputOffset + i ] = i;
	}

	__syncthreads();

	// TODO: sort pdDistanceMatrix and pdIndexMatrix
	// Currently doing externally using thrust. This method is so god awfully slow there is no point in even trying to optimise it.
	// Does it even make sense to sort them? All we want is the k lowest, surely this can be accomplished by sequentially scanning.
}

__global__ void KNNBruteForceCUDAKernel(	float3 const*	pdPosition,			// Agent positions.
											float *			pdDistanceMatrix,	// Global storage for distance matrix.
											size_t *		pdIndexMatrix,		// The indices which match postions in pdDistanceMatrix.
											size_t const	k,					// Number of neighbors to consider.
											size_t const	numAgents,			// Number of agents in the simulation.
											float3 const*	pdPositionOther,
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
	FLOAT3_GLOBAL_READ( shPosition, pdPosition );

	// TODO: This could be sped up by paging the read of pdPositionOther.

	// For each agent in the simulation...
	for( size_t i = 0; i < numOther; i++ )
	{
		if( groupWithSelf && index == i )
			continue;

		pdDistanceMatrix[ outputOffset + i ] = float3_distance( POSITION_SH( threadIdx.x ), pdPositionOther[ i ] );
		pdIndexMatrix[ outputOffset + i ] = i;
	}

	__syncthreads();
}
