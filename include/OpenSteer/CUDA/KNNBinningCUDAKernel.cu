#include "KNNBinningCUDA.cuh"

#include "../VectorUtils.cuh"

#include "CUDAKernelGlobals.cuh"

using namespace OpenSteer;

// Define the texture reference to access the appropriate bin_cell's index.
texture< uint, cudaTextureType3D, cudaReadModeElementType > texCellIndicesNormalized;
texture< uint, cudaTextureType3D, cudaReadModeElementType > texCellIndices;

// Constant memory used to hold the worldSize and worldCells values.
__constant__ float3		constWorldSize;
__constant__ float3		constWorldStep;
__constant__ float3		constWorldStepNormalized;
__constant__ uint3		constWorldCells;

// Fetch the cell index from texCellIndicesNormalized at a given world {x,y,z} position.
#define CELL_INDEX_NORMALIZED( pos )	( tex3D( texCellIndicesNormalized, pos.x, pos.y, pos.z ) )
// Fetch the cell index from texCellIndices at a given texel (x,y,z) coordinate.
#define CELL_INDEX( x, y, z )			( tex3D( texCellIndices, x, y, z ) )

// Kernel declarations.
extern "C"
{
	// Bind the textures to the input cudaArray.
	__host__ void KNNBinningCUDABindTexture( cudaArray * pCudaArray );
	// Unbind the textures.
	__host__ void KNNBinningCUDAUnbindTexture( void );

	// Use to precompute the neighbors of each cell once per decomposition.
	__global__ void KNNBinningComputeCellNeighbors2D(	bin_cell const*	pdCells,			// In:	Cell data.
														uint *			pdCellNeighbors,	// Out:	Array of computed cell neighbors.
														size_t const	neighborsPerCell,	// In:	Number of neighbors per cell.
														int const		radius,				// In:	Search radius.
														size_t const	numCells			// In:	Number of cells.
														);
													

	// Kernel to set initial bin indices of vehicles in the simulation.
	__global__ void KNNBinningBuildDB(		float3 const*	pdPosition,				// In:	Positions of each agent.
											size_t *		pdAgentIndices,			// Out:	Indices of each agent.
											size_t *		pdCellIndices,			// Out:	Indices of the cell each agent is in.
											size_t const	numAgents
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

	__global__ void KNNBinningKernel(		float3 const*	pdPositionSorted,			// In:	(sorted) Agent positions.

											uint const*		pdAgentIndices,				// In:	(sorted) Indices of each agent.
											uint const*		pdCellIndices,				// In:	(sorted) Indices of the cell each agent is currently in.

											uint const*		pdCellStart,				// In:	Start index of each cell in pdCellIndices.
											uint const*		pdCellEnd,					// In:	End index of each cell in pdCellIndices.

											uint const*		pdCellNeighbors,			// In:	Indices of the neighbors to radius distance of each cell.
											size_t const	neighborsPerCell,			// In:	Number of neighbors per cell in the pdCellNeighbors array.

											uint *			pdKNNIndices,				// Out:	Indices of K Nearest Neighbors in pdPosition.
											float *			pdKNNDistances,				// Out:	Distances of the K Nearest Neighbors in pdPosition.

											size_t const	k,							// In:	Number of neighbors to consider.
											size_t const	radius,						// In:	Maximum radius (in cells) to consider.
											size_t const	numAgents					// In:	Number of agents in the simulation.
											);
}

__global__ void KNNBinningComputeCellNeighbors2D(	bin_cell const*	pdCells,			// In:	Cell data.
													uint *			pdCellNeighbors,	// Out:	Array of computed cell neighbors.
													size_t const	neighborsPerCell,	// In:	Number of neighbors per cell.
													int const		radius,				// In:	Search radius.
													size_t const	numCells			// In:	Number of cells.
													)
{
	int const index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if( index >= numCells )
		return;

	__shared__ float3 shPosition[THREADSPERBLOCK];
	extern __shared__ uint shNeighboringCells[];
	
	// Read the position of this thread's cell to shared memory.
	shPosition[ threadIdx.x ] = pdCells[index].position;

	// Normalize the positions.
	POSITION_SH( threadIdx.x ).x = (POSITION_SH( threadIdx.x ).x + 0.5f * constWorldSize.x) / constWorldSize.x;
	POSITION_SH( threadIdx.x ).y = (POSITION_SH( threadIdx.x ).y + 0.5f * constWorldSize.y) / constWorldSize.y;
	POSITION_SH( threadIdx.x ).z = (POSITION_SH( threadIdx.x ).z + 0.5f * constWorldSize.z) / constWorldSize.z;

	__syncthreads();

	// Get the first cell index (radius 0).
	shNeighboringCells[ threadIdx.x * neighborsPerCell ] = CELL_INDEX_NORMALIZED( POSITION_SH( threadIdx.x ) );

	int i = 1;
	// Compute the start offset into shNeighboringCells for this radius.
	int offset = threadIdx.x * neighborsPerCell;

	// For increasing radius...
	for( int iCurrentRadius = 1; iCurrentRadius <= radius; iCurrentRadius++ )
	{
		for( int dz = -iCurrentRadius; dz <= iCurrentRadius; dz++ )
		{
			for( int dx = -iCurrentRadius; dx <= iCurrentRadius; dx++ )
			{
				// Only do for the outside cells.
				if( dz == -iCurrentRadius || dz == iCurrentRadius || dx == -iCurrentRadius || dx == iCurrentRadius )
				{
					float3 queryPosition = make_float3(	POSITION_SH( threadIdx.x ).x + dx * constWorldStepNormalized.x,
														POSITION_SH( threadIdx.x ).y,
														POSITION_SH( threadIdx.x ).z + dz * constWorldStepNormalized.z
														);

					uint cellIndex = CELL_INDEX_NORMALIZED( queryPosition );

					// Do not add duplicate cells.
					for( int iDup = 0; iDup < i; iDup++ )
					{
						if( shNeighboringCells[offset+iDup] == cellIndex )
							cellIndex = UINT_MAX;
					}

					shNeighboringCells[offset + i++] = cellIndex;
				}
			}
		}

	}

	__syncthreads();
	for( int i = 0; i < neighborsPerCell; i++ )
	{
		pdCellNeighbors[ index * neighborsPerCell + i ] = shNeighboringCells[ offset + i ];
	}
}

__host__ void KNNBinningCUDABindTexture( cudaArray * pdCudaArray )
{
	static cudaChannelFormatDesc const channelDesc = cudaCreateChannelDesc< uint >();

	texCellIndicesNormalized.normalized = true;
	texCellIndicesNormalized.filterMode = cudaFilterModePoint;
	// Clamp out of bounds coordinates to the edge of the texture.
	texCellIndicesNormalized.addressMode[0] = cudaAddressModeClamp;
	texCellIndicesNormalized.addressMode[1] = cudaAddressModeClamp;
	texCellIndicesNormalized.addressMode[2] = cudaAddressModeClamp;

	CUDA_SAFE_CALL( cudaBindTextureToArray( texCellIndicesNormalized, pdCudaArray, channelDesc ) );

	texCellIndices.normalized = false;
	texCellIndices.filterMode = cudaFilterModePoint;
	texCellIndices.addressMode[0] = cudaAddressModeClamp;
	texCellIndices.addressMode[1] = cudaAddressModeClamp;
	texCellIndices.addressMode[2] = cudaAddressModeClamp;

	CUDA_SAFE_CALL( cudaBindTextureToArray( texCellIndices, pdCudaArray, channelDesc ) );
}

__host__ void KNNBinningCUDAUnbindTexture( void )
{
	CUDA_SAFE_CALL( cudaUnbindTexture( texCellIndicesNormalized ) );
	CUDA_SAFE_CALL( cudaUnbindTexture( texCellIndices ) );
}

__global__ void KNNBinningBuildDB(	float3 const*	pdPosition,				// In:	Positions of each agent.
									size_t *		pdAgentIndices,			// Out:	Indices of each agent.
									size_t *		pdCellIndices,			// Out:	Indices of the cell each agent is in.
									size_t const	numAgents
									)
{
	// Offset of this agent in the global array.
	int const index = (blockIdx.x * blockDim.x) + threadIdx.x;

	// Check bounds.
	if( index >= numAgents )
		return;

	// Copy the positions to shared memory.
	__shared__ float3 shPosition[THREADSPERBLOCK];
	FLOAT3_GLOBAL_READ( shPosition, pdPosition );

	// Normalize the positions.
	POSITION_SH( threadIdx.x ).x = (POSITION_SH( threadIdx.x ).x + 0.5f * constWorldSize.x) / constWorldSize.x;
	POSITION_SH( threadIdx.x ).y = (POSITION_SH( threadIdx.x ).y + 0.5f * constWorldSize.y) / constWorldSize.y;
	POSITION_SH( threadIdx.x ).z = (POSITION_SH( threadIdx.x ).z + 0.5f * constWorldSize.z) / constWorldSize.z;
	
	// Write the agent's cell index out to global memory.
	pdCellIndices[index] = CELL_INDEX_NORMALIZED( POSITION_SH( threadIdx.x ) );

	// Write the agent's index out to global memory.
	pdAgentIndices[index] = index;
}


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
	int const index = (blockIdx.x * blockDim.x) + threadIdx.x;

	// Check bounds.
	if( index >= numAgents )
		return;

	__shared__ uint shCellIndices[THREADSPERBLOCK+1];

	// Shared memory so we can coalesce the writes of sorted data to global memory.
	__shared__ float3 shPositionSorted[THREADSPERBLOCK];
	__shared__ float3 shDirectionSorted[THREADSPERBLOCK];
	__shared__ float shSpeedSorted[THREADSPERBLOCK];

	// Read the cell index of this agent.
	uint iCellIndex = pdCellIndices[ index ];
	__syncthreads();
	
	// Store cell index data in shared memory so that we can look 
	// at the neighboring agent's value without two reads per thread.
	shCellIndices[ threadIdx.x + 1 ] = iCellIndex;

	if( index > 0 && threadIdx.x == 0 )
	{
		// First thread in block must load neighbor agent cell index.
		shCellIndices[0] = pdCellIndices[ index - 1 ];
	}

	__syncthreads();

	// If this agent has a different cell index to the previous
	// agent then it must be the first in the cell,
	// so store the index of this agent in the cell.
	// As it isn't the first agent, it must also be the cell end of
	// the previous particle's cell

	if( index == 0 || iCellIndex != shCellIndices[ threadIdx.x ] )
	{
		pdCellStart[ iCellIndex ] = index;
		if( index > 0 )
			pdCellEnd[ shCellIndices[ threadIdx.x ] ] = index;
	}

	// If this is the last agent, the end index for the cell will be index + 1
	if( index == (numAgents - 1) )
	{
		pdCellEnd[ iCellIndex ] = index + 1;
	}

	// Use the sorted index to reorder the position/direction/speed data.
	uint const iSortedIndex = pdAgentIndices[ index ];

	// TODO:	This is potentially a lot faster using texture memory for the input data due to the 'random' nature of the access.
	//			Will require the transition to float4 instead of float3 to store the data though.
	shPositionSorted[ threadIdx.x ] = pdPosition[ iSortedIndex ];
	shDirectionSorted[ threadIdx.x ] = pdDirection[ iSortedIndex ];
	shSpeedSorted[ threadIdx.x ] = pdSpeed[ iSortedIndex ];

	__syncthreads();

	// Write to global memory.
	pdSpeedSorted[ index ] = shSpeedSorted[ threadIdx.x ];
	__syncthreads();
	
	FLOAT3_GLOBAL_WRITE( pdPositionSorted, shPositionSorted );
	FLOAT3_GLOBAL_WRITE( pdDirectionSorted, shDirectionSorted );
}

__global__ void KNNBinningKernel(	float3 const*	pdPositionSorted,			// In:	(sorted) Agent positions.

									uint const*		pdAgentIndices,				// In:	(sorted) Indices of each agent.
									uint const*		pdCellIndices,				// In:	(sorted) Indices of the cell each agent is currently in.

									uint const*		pdCellStart,				// In:	Start index of each cell in pdCellIndices.
									uint const*		pdCellEnd,					// In:	End index of each cell in pdCellIndices.

									uint const*		pdCellNeighbors,			// In:	Indices of the neighbors to radius distance of each cell.
									size_t const	neighborsPerCell,			// In:	Number of neighbors per cell in the pdCellNeighbors array.

									uint *			pdKNNIndices,				// Out:	Indices of K Nearest Neighbors in pdPosition.
									float *			pdKNNDistances,				// Out:	Distances of the K Nearest Neighbors in pdPosition.

									size_t const	k,							// In:	Number of neighbors to consider.
									size_t const	radius,						// In:	Maximum radius (in cells) to consider.
									size_t const	numAgents					// In:	Number of agents in the simulation.
									)
{
	// Index of this agent.
	int const index = (blockIdx.x * blockDim.x) + threadIdx.x;

	// Check bounds.
	if( index >= numAgents )
		return;

	__shared__ float3 shPosition[THREADSPERBLOCK];

	// Shared memory for local priority queue computations.
	extern __shared__ uint shKNNIndices[];
	float * shKNNDistances = (float*)shKNNIndices + THREADSPERBLOCK * k;

	// Set all elements of shKNNDistances to FLT_MAX, shKNNIndices to UINT_MAX.
	for( uint i = 0; i < k; i++ )
	{
		shKNNIndices[(threadIdx.x * k) + i] = UINT_MAX;
		shKNNDistances[(threadIdx.x * k) + i] = FLT_MAX;
	}

	// Coalesce read the agent positions.
	FLOAT3_GLOBAL_READ( shPosition, pdPositionSorted );
	
	// Store this thread's agent index and cell index in registers. TODO: texture memory.
	uint const		agentIndex				= pdAgentIndices[ index ];
	uint			cellIndex				= pdCellIndices[ index ];

	// Get the offset for the neighbors of this cell.
	int const cellNeighborsOffset = cellIndex * neighborsPerCell;

	// For each of the neighbors of the current cell...
	for( int iCellNeighbor = 0; iCellNeighbor < neighborsPerCell; iCellNeighbor++ )
	{
		cellIndex = pdCellNeighbors[ cellNeighborsOffset + iCellNeighbor ];

		if( cellIndex == UINT_MAX )	// There is no neighboring cell in this position.
			continue;

		// For each agent in the cell...
		for( uint otherIndexSorted = pdCellStart[ cellIndex ]; otherIndexSorted < pdCellEnd[ cellIndex ]; otherIndexSorted++ )
		{
			// Get the index of the other agent (unsorted).
			uint const otherIndex = pdAgentIndices[ otherIndexSorted ];
			
			// Do not include self.
			if( agentIndex == otherIndex )
				continue;

			// Compute the distance between this agent and the one at i.
			// TODO: texture memory....
			float const dist = float3_distance( POSITION_SH( threadIdx.x ), pdPositionSorted[ otherIndexSorted ] );

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
	}

	__syncthreads();

	// Write the shKNNIndices and shKNNDistances values out to global memory.
	for( uint i = 0; i < k; i++ )
	{
		pdKNNIndices[agentIndex*k + i] = shKNNIndices[threadIdx.x*k + i];
		pdKNNDistances[agentIndex*k + i] = shKNNDistances[threadIdx.x*k + i];
	}
	__syncthreads();
}
