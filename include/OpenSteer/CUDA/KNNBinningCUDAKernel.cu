#include "KNNBinningCUDA.cuh"

#include "../VectorUtils.cuh"

#include "CUDAKernelGlobals.cuh"

using namespace OpenSteer;

// Define the texture reference to access the appropriate bin_cell's index.
texture< uint, cudaTextureType3D, cudaReadModeElementType > texCellIndices;

// Fetch the bin from texBinCells at a given world {x,y,z} position.
#define CELLINDEX( pos ) ( tex3D( texCellIndices, pos.x, pos.z, pos.y ) )

// Kernel declarations.
extern "C"
{
	// Bind texCellIndices to the cudaArray.
	__host__ void KNNBinningCUDABindTexture( cudaArray * pCudaArray );
	// Unbind the texture.
	__host__ void KNNBinningCUDAUnbindTexture( void );

	// Kernel to set initial bin indices of vehicles in the simulation.
	__global__ void KNNBinningBuildDB(	float3 const*	pdPosition,				// In:	Positions of each vehicle.
										uint *			pdAgentIndices,			// Out:	Indices of each vehicle.
										uint *			pdAgentCellIndices,		// Out:	Indices of the bin each vehicle is in.
										size_t const	numAgents				// In:	Number of agents in the simulation.
										);

	__global__ void KNNBinningReorderData(	float3 const*	pdPosition,			// In: Agent positions.
											float3 const*	pdDirection,		// In: Agent directions.
											float const*	pdSpeed,			// In: Agent speeds.
					
											uint const*		pdAgentIndices,		// In: (sorted) agent index.
											uint const*		pdCellIndices,		// In: (sorted) cell index agent is in.

											float3 *		pdPositionSorted,	// Out: Sorted agent positions.
											float3 *		pdDirectionSorted,	// Out: Sorted agent directions.
											float *			pdSpeedSorted,		// Out: Sorted agent speeds.

											uint *			pdCellStart,		// Out: Start index of this cell in pdCellIndices.
											uint *			pdCellEnd,			// Out: End index of this cell in pdCellIndices.

											size_t const	numAgents
											);

	__global__ void KNNBinningKernel(	float3 const*	pdPosition,			// In: Agent positions.
										size_t *		pdAgentIndices,		// In: (sorted) indices of each agent.
										size_t *		pdAgentCellIndices,	// In: (sorted) indices of the cell each agent is in.
										size_t const	k,					// In: Number of neighbors to consider.
										size_t const	radius,				// In: Maximum radius (in cells) to consider.
										size_t const	numAgents,			// In: Number of agents in the simulation.

										uint *			pdKNNIndices,		// Out: indices of K Nearest Neighbors in pdPosition.
										float *			pdKNNDistances		// Out: distances of the K Nearest Neighbors in pdPosition.
										);
}

//__global__ void KNNBinningKernel(	float3 const*	pdPosition,			// In: Agent positions.
//									uint *			pdKNNIndices,		// Out: indices of K Nearest Neighbors in pdPosition.
//									float *			pdKNNDistances,		// Out: distances of the K Nearest Neighbors in pdPosition.
//									size_t const	k,					// In: Number of neighbors to consider.
//									size_t const	radius,				// In: Maximum radius (in cells) to consider.
//									size_t const	numAgents,			// In: Number of agents in the simulation.
//									)

__global__ void KNNBinningReorderData(	float3 const*	pdPosition,			// In: Agent positions.
										float3 const*	pdDirection,		// In: Agent directions.
										float const*	pdSpeed,			// In: Agent speeds.
				
										uint const*		pdAgentIndices,		// In: (sorted) agent index.
										uint const*		pdCellIndices,		// In: (sorted) cell index agent is in.

										float3 *		pdPositionSorted,	// Out: Sorted agent positions.
										float3 *		pdDirectionSorted,	// Out: Sorted agent directions.
										float *			pdSpeedSorted,		// Out: Sorted agent speeds.

										uint *			pdCellStart,		// Out: Start index of this cell in pdCellIndices.
										uint *			pdCellEnd,			// Out: End index of this cell in pdCellIndices.

										size_t const	numAgents
										)
{
	// Offset of this agent.
	int offset = (blockIdx.x * blockDim.x) + threadIdx.x;

	// Check bounds.
	if( offset > numAgents )
		return;

	__shared__ uint shCellIndices[THREADSPERBLOCK+1];

	// Shared memory so we can coalesce the writes of sorted data back to global memory.
	__shared__ float3 shPositionSorted[THREADSPERBLOCK];
	__shared__ float3 shDirectionSorted[THREADSPERBLOCK];

	// Read the cell index of this agent.
	uint iCellIndex = pdCellIndices[offset];
	
	// Store cell index data in shared memory so that we can look 
	// at the neighboring agent's value without two reads per thread.
	shCellIndices[threadIdx.x+1] = iCellIndex;

	if( offset > 0 && threadIdx.x == 0 )
	{
		// First thread in block must load neighbor agent cell index.
		shCellIndices[0] = pdCellIndices[offset-1];
	}

	__syncthreads();

	// If this agent has a different cell index to the previous
	// agent then it must be the first in the cell,
	// so store the index of this agent in the cell.
	// As it isn't the first agent, it must also be the cell end of
	// the previous particle's cell

	if( offset == 0 || iCellIndex != shCellIndices[ threadIdx.x ] )
	{
		pdCellStart[iCellIndex] = offset;
		if( offset > 0 )
			pdCellEnd[ shCellIndices[ threadIdx.x ] ] = offset;
	}

	// If this is the last agent, the end index for the cell will be offset + 1
	if( offset == (numAgents - 1) )
	{
		pdCellEnd[ iCellIndex ] = offset + 1;
	}

	// Use the sorted index to reorder the position/direction/speed data.
	uint const iSortedIndex = pdAgentIndices[ offset ];
	shPositionSorted[ threadIdx.x ] = pdPosition[ iSortedIndex ];
	shDirectionSorted[ threadIdx.x ] = pdDirection[ iSortedIndex ];
	
	// Write to global memory.
	pdSpeedSorted[ offset ] = pdSpeed[ iSortedIndex ];
	FLOAT3_COALESCED_WRITE( pdPositionSorted, shPositionSorted );
	FLOAT3_COALESCED_WRITE( pdPositionSorted, shDirectionSorted );
}

__host__ void KNNBinningCUDABindTexture( cudaArray * pdCudaArray )
{
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<uint>();

	texCellIndices.normalized = true;
	texCellIndices.filterMode = cudaFilterModePoint;
	texCellIndices.addressMode[0] = cudaAddressModeClamp;
	texCellIndices.addressMode[1] = cudaAddressModeClamp;
	texCellIndices.addressMode[2] = cudaAddressModeClamp;

	CUDA_SAFE_CALL( cudaBindTextureToArray( texCellIndices, pdCudaArray, channelDesc ) );
}

__host__ void KNNBinningCUDAUnbindTexture( void )
{
	CUDA_SAFE_CALL( cudaUnbindTexture( texCellIndices ) );
}

__global__ void KNNBinningBuildDB(	float3 const*	pdPosition,				// In:	Positions of each vehicle.
									size_t *		pdAgentIndices,			// Out:	Indices of each vehicle.
									size_t *		pdAgentBinIndices,		// Out:	Indices of the bin each vehicle is in.
									size_t const	numAgents				// In:	Number of agents in the simulation.
									)
{
	// Offset of this agent in the global array.
	int offset = (blockIdx.x * blockDim.x) + threadIdx.x;

	// Check bounds.
	if( offset >= numAgents )
		return;

	// Copy the positions to shared memory.
	__shared__ float3 shPosition[THREADSPERBLOCK];
	FLOAT3_COALESCED_READ( shPosition, pdPosition );
	//POSITION_SH( threadIdx.x ) = POSITION( offset );

	// Write the agent's cell index out to global memory.
	pdAgentBinIndices[offset] = CELLINDEX( POSITION_SH( threadIdx.x ) );

	// Write the agent's index out to global memory.
	pdAgentIndices[offset] = offset;

	__syncthreads();
}