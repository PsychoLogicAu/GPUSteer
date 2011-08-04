#include "KNNBruteForceCUDA.cuh"

#include "../VehicleGroupData.cuh"
#include "../VectorUtils.cuh"

#include "CUDAKernelGlobals.h"

using namespace OpenSteer;

extern "C"
{
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
}