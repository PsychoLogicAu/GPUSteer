#include "KNNBruteForceCUDA.cuh"

//#include "../VehicleGroupData.cuh"
#include "../VectorUtils.cuh"

#include "CUDAKernelGlobals.h"

// For FLT_MAX.
#include "float.h"

using namespace OpenSteer;

extern "C"
{
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
	extern __shared__ float	shDist[];
	extern __shared__ uint	shInd[];

	// Set all elements of shDist to FLT_MAX.
	for( uint i = 0; i < k; i++ )
		shDist[threadIdx.x + i] = FLT_MAX;

	// TODO: Do we have enough shared memory to do this? Use registers?
	__shared__ float3 shPosition[THREADSPERBLOCK];
	POSITION_SH( threadIdx.x ) = POSITION( offset );

	// For each of the agents...
	for( uint i = 0; i < numAgents; i++ )
	{
		/*	// Test this... will likely be slower than computing k+1 and discarding the shortest...
		if( i == offset )
			continue;
		*/

		// Compute the distance between this agent and the one at i.
		float dist = float3_distance( POSITION_SH( threadIdx.x ), pdPosition[i] );

		if( shDist[threadIdx.x + (k - 1)] > dist )	// Distance of the kth closest agent.
		{
			// Set the distance and index.
			shDist[threadIdx.x + (k - 1)] = dist;
			shInd[threadIdx.x + (k - 1)] = i;

			// Bubble the values up...
			for( int slot = (k - 2); slot >= 0; slot-- )
			{
				if( shDist[threadIdx.x + slot] > shDist[threadIdx.x + (slot + 1)] )
				{
					swap( shDist[threadIdx.x + slot], shDist[threadIdx.x + (slot + 1)] );
					swap( shInd[threadIdx.x + slot], shInd[threadIdx.x + (slot + 1)] );
				}
			}
		}
	}

	// Write the shDist values out to global memory (TODO: coalesce the writes!).
	for( uint i = 0; i < k; i++ )
	{
		pdKNNIndices[offset + i] = shInd[threadIdx.x + i];
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
	// Currently doing externally using thrust.
	// Does it even make sense to sort them? All we want is the k lowest, surely this can be accomplished by sequentially scanning.
}