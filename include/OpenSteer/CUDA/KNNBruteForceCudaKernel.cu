#include "KNNBruteForceCUDA.cuh"

//#include "../VehicleGroupData.cuh"
#include "../VectorUtils.cuh"

#include "CUDAKernelGlobals.h"

// For FLT_MAX.
#include "float.h"

using namespace OpenSteer;

extern "C"
{
	// O(N2) time and memory approach.
	__global__ void KNNBruteForceCUDAKernel(	float3 const*	pdPosition,			// Agent positions.
												float *			pdDistanceMatrix,	// Global storage for distance matrix.
												size_t *		pdIndexMatrix,		// The indices which match postions in pdDistanceMatrix.
												//size_t *		pdKNNIndices,		// Output, indices of K Nearest Neighbors in pdPosition.
												size_t const	k,					// Number of neighbors to consider.
												size_t const	numAgents			// Number of agents in the simulation.
											);

	__global__ void KNNBruteForceCUDAKernelV2(	float3 const*	pdPosition,			// Agent positions.
												uint *			pdKNNIndices,		// Output, indices of K Nearest Neighbors in pdPosition.
												size_t const	k,					// Number of neighbors to consider.
												size_t const	numAgents			// Number of agents in the simulation.
											);

	__global__ void KNNBruteForceCUDAKernelV3(	float3 const*	pdPosition,			// Agent positions.
												uint *			pdKNNIndices,		// Output, indices of K Nearest Neighbors in pdPosition.
												size_t const	k,					// Number of neighbors to consider.
												size_t const	numAgents			// Number of agents in the simulation.
											);
	
}

__global__ void KNNBruteForceCUDAKernelV3(	float3 const*	pdPosition,			// Agent positions.
											uint *			pdKNNIndices,		// Output, indices of K Nearest Neighbors in pdPosition.
											size_t const	k,					// Number of neighbors to consider.
											size_t const	numAgents			// Number of agents in the simulation.
										)
{
	int offset = (blockIdx.x * blockDim.x) + threadIdx.x;

	// Check bounds.
	if( offset >= numAgents )
		return;

	// Store this thread's agent position in registers.
	float3 position = POSITION( offset );

	// Shared memory for local computations.
	extern __shared__ float shDist[];					// First half will be treated as the distance values.
	uint * shInd = (uint*)shDist + blockDim.x * k;		// Second half will be treated as the index values.

	// Set all elements of shDist to FLT_MAX.
	for( uint i = 0; i < k; i++ )
		shDist[(threadIdx.x * k) + i] = FLOAT_MAX;

	// Set the seeding values from the previous update.
	for( uint i = 0; i < k; i++ )
	{
		// Get the index of the ith closest agent from the last frame.
		uint const ind = pdKNNIndices[(blockIdx.x * blockDim.x) + (threadIdx.x + i)];

		// Compute the distance between this agent and the one at index.
		float const dist = float3_distance( position, POSITION( ind ) );

		if( dist < shDist[(threadIdx.x * k) + (k - 1)] )	// Distance of the kth closest agent.
		{
			// Agent at index i is the new kth closest. Set the distance and index in shared mem.
			shDist[(threadIdx.x * k) + (k - 1)] = dist;
			shInd[(threadIdx.x * k) + (k - 1)] = ind;

			// Bubble the values up...
			for( int slot = k - 2; slot >= 0; slot-- )
			{
				if( shDist[(threadIdx.x * k) + slot] > shDist[(threadIdx.x * k) + (slot + 1)] )
				{
					swap( shDist[(threadIdx.x * k) + slot], shDist[(threadIdx.x * k) + (slot + 1)] );
					swap( shInd[(threadIdx.x * k) + slot], shInd[(threadIdx.x * k) + (slot + 1)] );
				}
			}
		}
	}

	__syncthreads();

	// Compute the distance to each of the agents indexed by shInd for this agent.
	for( uint i = 0; i < k; i++ )
	{
		// Compute the distance between this agent and the one at threadIdx.x
		float const dist = float3_distance( position, POSITION( shInd[threadIdx.x * i] ) );

		if( shDist[(threadIdx.x * k) + (k - 1)] > dist )	// Distance of the kth closest agent.
		{
			// Agent at index i is the new kth closest. Set the distance and index in shared mem.
			shDist[(threadIdx.x * k) + (k - 1)] = dist;
			shInd[(threadIdx.x * k) + (k - 1)] = i;

			// Bubble the values up...
			for( int slot = k - 2; slot >= 0; slot-- )
			{
				if( shDist[(threadIdx.x * k) + slot] > shDist[(threadIdx.x * k) + (slot + 1)] )
				{
					swap( shDist[(threadIdx.x * k) + slot], shDist[(threadIdx.x * k) + (slot + 1)] );
					swap( shInd[(threadIdx.x * k) + slot], shInd[(threadIdx.x * k) + (slot + 1)] );
				}
			}
		}
	}

	__syncthreads();


	// For each of the agents...
	for( uint i = 0; i < numAgents; i++ )
	{
		// Test this... will likely be slower than computing k+1 and discarding the shortest...
		if( i == offset )
			continue;
		

		// Compute the distance between this agent and the one at i.
		float const dist = float3_distance( position, POSITION( i ) );

		if( shDist[(threadIdx.x * k) + (k - 1)] > dist )	// Distance of the kth closest agent.
		{
			// Agent at index i is the new kth closest. Set the distance and index in shared mem.
			shDist[(threadIdx.x * k) + (k - 1)] = dist;
			shInd[(threadIdx.x * k) + (k - 1)] = i;

			// Bubble the values up...
			for( int slot = k - 2; slot >= 0; slot-- )
			{
				if( shDist[(threadIdx.x * k) + slot] > shDist[(threadIdx.x * k) + (slot + 1)] )
				{
					swap( shDist[(threadIdx.x * k) + slot], shDist[(threadIdx.x * k) + (slot + 1)] );
					swap( shInd[(threadIdx.x * k) + slot], shInd[(threadIdx.x * k) + (slot + 1)] );
				}
			}
		}
	}
	//__syncthreads();
	//// Write the shDist values out to global memory (TODO: coalesce the writes!).
	//for( uint i = 0; i < k; i++ )
	//{
	//	pdKNNIndices[offset + i] = shInd[threadIdx.x + i];
	//}
	__syncthreads();
	// This should be the coalesced version of the above...
	for( uint i = 0; i < k; i++ )
	{
		pdKNNIndices[blockIdx.x * blockDim.x + threadIdx.x * i] = shInd[threadIdx.x * i];
	}
	//__syncthreads();
}

__global__ void KNNBruteForceCUDAKernelV2(	float3 const*	pdPosition,			// Agent positions.
											uint *			pdKNNIndices,		// Output, indices of K Nearest Neighbors in pdPosition.
											size_t const	k,					// Number of neighbors to consider.
											size_t const	numAgents			// Number of agents in the simulation.
										)
{
	int offset = (blockIdx.x * blockDim.x) + threadIdx.x;

	// Check bounds.
	if( offset >= numAgents )
		return;

	// Shared memory for local computations.
	extern __shared__ float shDist[];					// First half will be treated as the distance values.
	uint * shInd = (uint*)shDist + blockDim.x * k;		// Second half will be treated as the index values.

	// Set all elements of shDist to FLT_MAX and shInd to UINT_MAX.
	for( uint i = 0; i < k; i++ )
	{
		shDist[(threadIdx.x * k) + i] = FLOAT_MAX;
		shInd[(threadIdx.x * k) + i] = UINT_MAX;
	}

	// Store the positions locally.
	__shared__ float3 shPosition[THREADSPERBLOCK];
	POSITION_SH( threadIdx.x ) = POSITION( offset );

	__syncthreads();

	// For each of the agents...
	for( uint i = 0; i < numAgents; i++ )
	{
		// Test this... will likely be slower than computing k+1 and discarding the shortest...
		if( i == offset )
			continue;
		

		// Compute the distance between this agent and the one at i.
		float const dist = float3_distance( POSITION_SH( threadIdx.x ), POSITION( i ) );

		if( shDist[(threadIdx.x * k) + (k - 1)] > dist )	// Distance of the kth closest agent.
		{
			// Agent at index i is the new kth closest. Set the distance and index in shared mem.
			shDist[(threadIdx.x * k) + (k - 1)] = dist;
			shInd[(threadIdx.x * k) + (k - 1)] = i;

			// Bubble the values up...
			for( int slot = k - 2; slot >= 0; slot-- )
			{
				if( shDist[(threadIdx.x * k) + slot] > shDist[(threadIdx.x * k) + (slot + 1)] )
				{
					swap( shDist[(threadIdx.x * k) + slot], shDist[(threadIdx.x * k) + (slot + 1)] );
					swap( shInd[(threadIdx.x * k) + slot], shInd[(threadIdx.x * k) + (slot + 1)] );
				}
			}
		}
	}
	//__syncthreads();
	//// Write the shDist values out to global memory (TODO: coalesce the writes!).
	//for( uint i = 0; i < k; i++ )
	//{
	//	pdKNNIndices[offset + i] = shInd[threadIdx.x + i];
	//}
	__syncthreads();
	// This should be the coalesced version of the above...
	for( uint i = 0; i < k; i++ )
	{
		pdKNNIndices[blockIdx.x * blockDim.x + threadIdx.x * i] = shInd[threadIdx.x * i];
	}
	//__syncthreads();
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
	// Currently doing externally using thrust.
	// Does it even make sense to sort them? All we want is the k lowest, surely this can be accomplished by sequentially scanning.
}