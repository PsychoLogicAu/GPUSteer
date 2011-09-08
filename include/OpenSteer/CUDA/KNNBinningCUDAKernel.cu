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
//__constant__ float3		constWorldMin;
//__constant__ float3		constWorldMax;

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

	// Kernel to set initial bin indices of vehicles in the simulation.
	__global__ void KNNBinningBuildDB(	float3 const*	pdPosition,				// In:	Positions of each agent.
										float3 *		pdPositionNormalized,	// Out:	Normalized positions of each agent.
										size_t *		pdAgentIndices,			// Out:	Indices of each agent.
										size_t *		pdCellIndices,			// Out:	Indices of the cell each agent is in.
										size_t const	numAgents				// In:	Number of agents in the simulation.
										);

	__global__ void KNNBinningReorderData(	float3 const*	pdPosition,					// In: Agent positions.
											float3 const*	pdPositionNormalized,		// In:	Normalized agent positions.
											float3 const*	pdDirection,				// In: Agent directions.
											float const*	pdSpeed,					// In: Agent speeds.
					
											uint const*		pdAgentIndices,				// In: (sorted) agent index.
											uint const*		pdCellIndices,				// In: (sorted) cell index agent is in.

											float3 *		pdPositionSorted,			// Out: Sorted agent positions.
											float3 *		pdPositionNormalizedSorted,	// Out:	Sorted normalized agent positions.
											float3 *		pdDirectionSorted,			// Out: Sorted agent directions.
											float *			pdSpeedSorted,				// Out: Sorted agent speeds.

											uint *			pdCellStart,				// Out: Start index of this cell in pdCellIndices.
											uint *			pdCellEnd,					// Out: End index of this cell in pdCellIndices.

											size_t const	numAgents
											);

	__global__ void KNNBinningKernel(	float3 const*	pdPositionSorted,			// In:	Sorted agent positions.
										float3 const*	pdPositionNormalizedSorted,	// In:	Sorted normalized agent positions.

										uint const*		pdAgentIndices,				// In:	Sorted agent indices.
										uint const*		pdCellIndices,				// In:	Sorted indices of the cell each agent is currently in.
									
										uint const*		pdCellStart,				// In:	Start index of each cell in pdCellIndices.
										uint const*		pdCellEnd,					// In:	End index of each cell in pdCellIndices.

										uint *			pdKNNIndices,				// Out:	Indices of K Nearest Neighbors in pdPosition.
										float *			pdKNNDistances,				// Out:	Distances of the K Nearest Neighbors in pdPosition.

										size_t const	k,							// In:	Number of neighbors to consider.
										size_t const	radius,						// In:	Maximum radius (in cells) to consider.
										size_t const	numAgents					// In:	Number of agents in the simulation.
										);
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

//__inline__ __device__ bool WithinBounds( float3 const* point )
//{
//	return !(	point->x < constWorldMin.x || point->x > constWorldMax.x ||
//				point->y < constWorldMin.y || point->y > constWorldMax.y ||
//				point->z < constWorldMin.z || point->z > constWorldMax.z
//				);
//}

__device__ void GetCellsWithinRadius3D(	float3 const&	positionNormalized,	// In:	Normalized position to find neighboring cells for.
										uint *			pdCells,			// Out:	Array of cell indices.
										int const&		r					// In:	Radius.
													)
{
	for( int dz = -r, iz = 0; dz <= r; dz++, iz++ )
	{
		for( int dy = -r, iy = 0; dy <= r; dy++, iy++ )
		{
			for( int dx = -r, ix = 0; dx <= r; dx++, ix++ )
			{
				int const index = iz*iy*(r*2+1) + iy*(r*2+1) + ix;
				pdCells[ index ] = UINT_MAX;

				// Compute the query position for this iteration.
				float3	queryPosition = make_float3(	positionNormalized.x + dx * constWorldStepNormalized.x,
														positionNormalized.y + dy * constWorldStepNormalized.y,
														positionNormalized.z + dz * constWorldStepNormalized.z
														);

				// Get the cell index for the query position.
				uint const cellIndex = CELL_INDEX_NORMALIZED( queryPosition );

				// Don't add the same cell twice. Will be a necessary check if the cell sizes are non-uniform.
				bool bContained = false;
				for( uint i = 0; i < index; i++ )
				{
					if( pdCells[i] == cellIndex )
						bContained = true;
				}
				if( bContained )
					continue;

				// Write the cell index to the output array.
				pdCells[ index ] = cellIndex;
			}
		}
	}
}

__device__ void GetCellsWithinRadius2D(	float3 const&	positionNormalized,	// In:	Position to find neighbors for.
										uint *			pdCells,			// Out:	Array of cell indices.
										int const&		r					// In:	Radius.
										)
{
	for( int dz = -r, iz = 0; dz <= r; dz++, iz++ )
	{
		for( int dx = -r, ix = 0; dx <= r; dx++, ix++ )
		{
			int const index = iz*(r*2+1) + ix;
			// Set the cell index to UINT_MAX.
			pdCells[ index ] = UINT_MAX;

			// Compute the query position for this iteration.
			float3	queryPosition = make_float3(	positionNormalized.x + dx * constWorldStepNormalized.x,
													positionNormalized.y,
													positionNormalized.z + dz * constWorldStepNormalized.z
													);

			// Get the cell index for the query position.
			uint const cellIndex = CELL_INDEX_NORMALIZED( queryPosition );

			// Don't add the same cell twice. Will be a necessary check if the cell sizes are non-uniform.
			bool bContained = false;
			for( uint i = 0; i < index; i++ )
			{
				if( pdCells[i] == cellIndex )
					bContained = true;
			}
			if( bContained )
				continue;

			// Write the cell index to the output array.
			pdCells[ index ] = cellIndex;
		}
	}
}

__global__ void KNNBinningBuildDB(	float3 const*	pdPosition,				// In:	Positions of each agent.
									float3 *		pdPositionNormalized,	// Out:	Normalized positions of each agent.
									size_t *		pdAgentIndices,			// Out:	Indices of each agent.
									size_t *		pdCellIndices,			// Out:	Indices of the cell each agent is in.
									size_t const	numAgents				// In:	Number of agents in the simulation.
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

	__syncthreads();

	// Normalize the positions.
	POSITION_SH( threadIdx.x ).x = (POSITION_SH( threadIdx.x ).x + 0.5f * constWorldSize.x) / constWorldSize.x;
	//POSITION_SH( threadIdx.x ).y = (POSITION_SH( threadIdx.x ).y + 0.5f * constWorldSize.y) / constWorldSize.y;	// TODO: effect of normalizing y?
	POSITION_SH( threadIdx.x ).z = (POSITION_SH( threadIdx.x ).z + 0.5f * constWorldSize.z) / constWorldSize.z;

	// Write the normalized positions to global memory.
	FLOAT3_GLOBAL_WRITE( pdPositionNormalized, shPosition );

	pdCellIndices[index] = tex3D( texCellIndicesNormalized, POSITION_SH( threadIdx.x ).x, POSITION_SH( threadIdx.x ).y, POSITION_SH( threadIdx.x ).z );
	__syncthreads();

	// Write the agent's index out to global memory.
	pdAgentIndices[index] = index;
	__syncthreads();
}


__global__ void KNNBinningReorderData(	float3 const*	pdPosition,					// In:	Agent positions.
										float3 const*	pdPositionNormalized,		// In:	Normalized agent positions.
										float3 const*	pdDirection,				// In:	Agent directions.
										float const*	pdSpeed,					// In:	Agent speeds.
				
										uint const*		pdAgentIndices,				// In:	(sorted) agent index.
										uint const*		pdCellIndices,				// In:	(sorted) cell index agent is in.

										float3 *		pdPositionSorted,			// Out:	Sorted agent positions.
										float3 *		pdPositionNormalizedSorted,	// Out:	Sorted agent positions.
										float3 *		pdDirectionSorted,			// Out:	Sorted agent directions.
										float *			pdSpeedSorted,				// Out:	Sorted agent speeds.

										uint *			pdCellStart,				// Out:	Start index of this cell in pdCellIndices.
										uint *			pdCellEnd,					// Out:	End index of this cell in pdCellIndices.

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
	__shared__ float3 shPositionNormalizedSorted[THREADSPERBLOCK];
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
	shPositionNormalizedSorted[ threadIdx.x ] = pdPositionNormalized[ iSortedIndex ];

	__syncthreads();

	// Write to global memory.
	pdSpeedSorted[ index ] = shSpeedSorted[ threadIdx.x ];
	__syncthreads();
	
	FLOAT3_GLOBAL_WRITE( pdPositionSorted, shPositionSorted );
	FLOAT3_GLOBAL_WRITE( pdDirectionSorted, shDirectionSorted );
	FLOAT3_GLOBAL_WRITE( pdPositionNormalizedSorted, shPositionNormalizedSorted );
}

__global__ void KNNBinningKernel(	float3 const*	pdPositionSorted,			// In:	(sorted) Agent positions.
									float3 const*	pdPositionNormalizedSorted,

									uint const*		pdAgentIndices,				// In:	(sorted) Indices of each agent.
									uint const*		pdCellIndices,				// In:	(sorted) Indices of the cell each agent is currently in.
								
									uint const*		pdCellStart,				// In:	Start index of each cell in pdCellIndices.
									uint const*		pdCellEnd,					// In:	End index of each cell in pdCellIndices.

									uint *			pdKNNIndices,				// Out:	Indices of K Nearest Neighbors in pdPosition.
									float *			pdKNNDistances,				// Out:	Distances of the K Nearest Neighbors in pdPosition.

									size_t const	k,							// In:	Number of neighbors to consider.
									size_t const	radius,						// In:	Maximum radius (in cells) to consider.
									size_t const	numAgents					// In:	Number of agents in the simulation.
									)
{
	// Offset of this agent.
	int const index = (blockIdx.x * blockDim.x) + threadIdx.x;

	// Check bounds.
	if( index >= numAgents )
		return;

	// Shared memory for local priority queue computations.
	extern __shared__ uint shKNNIndices[];
	float * shKNNDistances = (float*)shKNNIndices + THREADSPERBLOCK * k;

	// Set all elements of shKNNDistances to FLT_MAX, shKNNIndices to UINT_MAX.
	for( uint i = 0; i < k; i++ )	// TODO: test effect of unrolling.
	{
		shKNNIndices[(threadIdx.x * k) + i] = UINT_MAX;
		shKNNDistances[(threadIdx.x * k) + i] = FLT_MAX;
	}

	// Store this thread's agent index and cell index in registers.
	uint const		agentIndex = pdAgentIndices[ index ];
	//uint const		cellIndex = pdCellIndices[ index ];
	float3 const	agentPosition = pdPositionSorted[ index ];
	float3 const	agentPositionNormalized = pdPositionNormalizedSorted[ index ];

	//
	// TODO: for each surrounding cell within radius...
	//

	// TODO: support for varying radius.
	__shared__ uint shNeighboringCells[9*THREADSPERBLOCK];
	GetCellsWithinRadius2D( agentPositionNormalized, &shNeighboringCells[threadIdx.x*9], radius );

	uint cellIndex = UINT_MAX;
	for( uint i = 0; i < 9; i++ )
	{
		cellIndex = shNeighboringCells[threadIdx.x*9+i];
		if( UINT_MAX == cellIndex )
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
			float const dist = float3_distance( agentPosition, pdPositionSorted[ otherIndexSorted ] );

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
