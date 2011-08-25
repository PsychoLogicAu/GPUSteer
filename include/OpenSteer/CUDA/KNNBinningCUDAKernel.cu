#include "KNNBinningCUDA.cuh"

#include "../VectorUtils.cuh"

#include "CUDAKernelGlobals.cuh"

using namespace OpenSteer;

// Define the texture reference to access the appropriate bin_cell's index.
texture< uint, cudaTextureType3D, cudaReadModeElementType > texCellIndicesNormalized;
texture< uint, cudaTextureType3D, cudaReadModeElementType > texCellIndices;

// Fetch the cell index from texCellIndicesNormalized at a given world {x,y,z} position.
#define CELL_INDEX_NORMALIZED( pos )	( tex3D( texCellIndicesNormalized, pos.x, pos.z, pos.y ) )
// Fetch the cell index from texCellIndices at a given texel (x,y,z) coordinate.
#define CELL_INDEX( x, y, z )			( tex3D( texCellIndices, x, z, y ) )

// Kernel declarations.
extern "C"
{
	// Bind the textures to the input cudaArray.
	__host__ void KNNBinningCUDABindTexture( cudaArray * pCudaArray );
	// Unbind the textures.
	__host__ void KNNBinningCUDAUnbindTexture( void );

	// Kernel to set initial bin indices of vehicles in the simulation.
	__global__ void KNNBinningBuildDB(	float3 const*	pdPosition,				// In:	Positions of each agent.
										size_t *		pdAgentIndices,			// Out:	Indices of each agent.
										size_t *		pdCellIndices,			// Out:	Indices of the cell each agent is in.
										size_t const	numAgents,				// In:	Number of agents in the simulation.
										float3 const	worldSize				// In:	Extents of the world (for normalizing the positions).
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

	__global__ void KNNBinningKernel(	float3 const*	pdPositionSorted,	// In:	(sorted) Agent positions.

										uint const*		pdAgentIndices,		// In:	(sorted) Indices of each agent.
										uint const*		pdCellIndices,		// In:	(sorted) Indices of the cell each agent is in.
									
										uint const*		pdCellStart,		// In:	Start index of each cell in pdCellIndices.
										uint const*		pdCellEnd,			// In:	End index of each cell in pdCellIndices.

										uint *			pdKNNIndices,		// Out:	Indices of K Nearest Neighbors in pdPosition.
										float *			pdKNNDistances,		// Out:	Distances of the K Nearest Neighbors in pdPosition.

										size_t const	k,					// In:	Number of neighbors to consider.
										size_t const	radius,				// In:	Maximum radius (in cells) to consider.
										size_t const	numAgents			// In:	Number of agents in the simulation.
										);
}

__host__ void KNNBinningCUDABindTexture( cudaArray * pdCudaArray )
{
	static cudaChannelFormatDesc const channelDesc = cudaCreateChannelDesc< uint >();

	texCellIndicesNormalized.normalized = true;
	texCellIndicesNormalized.filterMode = cudaFilterModePoint;
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


__global__ void KNNBinningKernel(	float3 const*	pdPositionSorted,	// In:	(sorted) Agent positions.

									uint const*		pdAgentIndices,		// In:	(sorted) Indices of each agent.
									uint const*		pdCellIndices,		// In:	(sorted) Indices of the cell each agent is in.
								
									uint const*		pdCellStart,		// In:	Start index of each cell in pdCellIndices.
									uint const*		pdCellEnd,			// In:	End index of each cell in pdCellIndices.

									uint *			pdKNNIndices,		// Out:	Indices of K Nearest Neighbors in pdPosition.
									float *			pdKNNDistances,		// Out:	Distances of the K Nearest Neighbors in pdPosition.

									size_t const	k,					// In:	Number of neighbors to consider.
									size_t const	radius,				// In:	Maximum radius (in cells) to consider.
									size_t const	numAgents			// In:	Number of agents in the simulation.
									)
{
	// Offset of this agent.
	int const offset = (blockIdx.x * blockDim.x) + threadIdx.x;

	// Check bounds.
	if( offset >= numAgents )
		return;

	// Shared memory for local priority queue computations.
	extern __shared__ float shDist[];				// First half will be treated as the distance values.
	uint * shInd = (uint*)shDist + blockDim.x * k;	// Second half will be treated as the index values.

	// Set all elements of shDist to FLT_MAX, shInd to UINT_MAX.
	for( uint i = 0; i < k; i++ )
	{
		shDist[(threadIdx.x * k) + i] = FLT_MAX;
		shInd[(threadIdx.x * k) + i] = UINT_MAX;
	}

	
	// Store this thread's agent position & cell in registers.
	float3 const	position = pdPositionSorted[ offset ];
	uint const		cell = pdCellIndices[ offset ];

	// TODO: for each surrounding cell within radius...

	// For each agent in the cell...
	for( uint i = pdCellStart[ cell ]; i < pdCellEnd[ cell ]; i++ )
	{
		uint agentIndex = pdAgentIndices[ i ];
		
		if( offset == agentIndex )
			continue;

		// Compute the distance between this agent and the one at i.
		float const dist = float3_distance( position, pdPositionSorted[ agentIndex ] );

		if( dist < shDist[(threadIdx.x * k) + (k - 1)] )	// Distance of the kth closest agent.
		{
			// Agent at index i is the new (at least) kth closest. Set the distance and index in shared mem.
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
				else
					break;
			}
		}
	}

	__syncthreads();

	// Write the shInd and shDist values out to global memory (TODO: coalesce the writes!).
	for( uint i = 0; i < k; i++ )
	{
		pdKNNIndices[offset + i] = shInd[threadIdx.x + i];
		pdKNNDistances[offset + i] = shDist[threadIdx.x + i];
	}
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
	int const offset = (blockIdx.x * blockDim.x) + threadIdx.x;

	// Check bounds.
	if( offset >= numAgents )
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

	__syncthreads();
	
	// Write to global memory.
	pdSpeedSorted[ offset ] = pdSpeed[ iSortedIndex ];
	FLOAT3_COALESCED_WRITE( pdPositionSorted, shPositionSorted );
	FLOAT3_COALESCED_WRITE( pdPositionSorted, shDirectionSorted );
}

__global__ void KNNBinningBuildDB(	float3 const*	pdPosition,				// In:	Positions of each agent.
									size_t *		pdAgentIndices,			// Out:	Indices of each agent.
									size_t *		pdCellIndices,			// Out:	Indices of the cell each agent is in.
									size_t const	numAgents,				// In:	Number of agents in the simulation.
									float3 const	worldSize				// In:	Extents of the world (for normalizing the positions).
									)
{
	// Offset of this agent in the global array.
	int offset = (blockIdx.x * blockDim.x) + threadIdx.x;

	// Check bounds.
	if( offset >= numAgents )
		return;

	// FIXME: too many are being hashed into cell 0...why?

	// Copy the positions to shared memory.
	__shared__ float3 shPosition[THREADSPERBLOCK];
	FLOAT3_COALESCED_READ( shPosition, pdPosition );

	__syncthreads();

	// Normalize the positions.
	POSITION_SH( threadIdx.x ) = make_float3(	(POSITION_SH( threadIdx.x ).x + 0.5f * worldSize.x) / worldSize.x, 
												(POSITION_SH( threadIdx.x ).y + 0.5f * worldSize.y) / worldSize.y,
												(POSITION_SH( threadIdx.x ).z + 0.5f * worldSize.z) / worldSize.z );

	//__syncthreads();

	// Write the agent's cell index out to global memory.
	pdCellIndices[offset] = CELL_INDEX_NORMALIZED( POSITION_SH( threadIdx.x ) );

	// Write the agent's index out to global memory.
	pdAgentIndices[offset] = offset;

	__syncthreads();
}
