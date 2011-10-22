#include "KNNBinningV1.cuh"

#include "../VectorUtils.cuh"

#include "CUDAKernelGlobals.cuh"

using namespace OpenSteer;

// Texture references.
texture< uint, cudaTextureType3D, cudaReadModeElementType >		texCellIndicesNormalized;
texture< uint, cudaTextureType3D, cudaReadModeElementType >		texCellIndices;
texture< float4, cudaTextureType1D, cudaReadModeElementType >	texPosition;

__constant__ float3		constWorldSizeV1;
__constant__ float3		constWorldStepV1;
__constant__ float3		constWorldStepNormalizedV1;
__constant__ uint3		constWorldCellsV1;

#define CELL_INDEX( pos )	( tex3D( texCellIndices, pos.x, pos.y, pos.z ) )	// <--- for choke point.
//#define CELL_INDEX( pos )	( tex3D( texCellIndices, pos.x, pos.z, pos.y ) )	// <--- for boids.

// Kernel declarations.
extern "C"
{
	// Bind the textures to the input cudaArray.
	__host__ void KNNBinningV1BindTexture(			cudaArray * pCudaArray );
	// Unbind the textures.
	__host__ void KNNBinningV1UnbindTexture( void );

	__host__ void KNNBinningV1KernelBindTextures(			uint const*		pdBCellStart,
														uint const*		pdBCellEnd,
														uint const*		pdBIndices,
														float4 const*	pdBPositionSorted,
														uint const		numCells,
														uint const		numB
														);

	__host__ void KNNBinningV1KernelUnbindTextures( void );
	__host__ void KNNBinningV1ReorderDBBindTextures(		float4 const*	pdPosition,
														uint const		numAgents
														);
	__host__ void KNNBinningV1ReorderDBUnbindTextures( void );

	// Kernel to set initial bin indices of vehicles in the simulation.
	__global__ void KNNBinningV1BuildDB(					float4 const*	pdPosition,				// In:	Positions of each agent.
														size_t *		pdAgentIndices,			// Out:	Indices of each agent.
														size_t *		pdCellIndices,			// Out:	Indices of the cell each agent is in.
														size_t const	numAgents
														);

	// Reorder the positions on pdCellIndices, and compute the cell start and end indices.
	__global__ void KNNBinningV1ReorderDB(				uint const*		pdAgentIndices,		// In: (sorted) agent index.
														uint const*		pdCellIndices,		// In: (sorted) cell index agent is in.

														float4 *		pdPositionSorted,	// Out: Sorted agent positions.

														uint *			pdCellStart,		// Out: Start index of this cell in pdCellIndices.
														uint *			pdCellEnd,			// Out: End index of this cell in pdCellIndices.

														size_t const	numAgents
														);

	__global__ void KNNBinningV1Kernel(					// Group A
														float4 const*	pdAPositionSorted,			// In:	Sorted group A positions.
														uint const*		pdAIndices,					// In:	Sorted group A indices
														uint const		numA,						// In:	Size of group A.

														// Cell neighbor info.
														int const		radius,						// In:	Search radius (in cells) to consider.

														// Output data.
														uint *			pdKNNIndices,				// Out:	Indices of K Nearest Neighbors in pdPosition.
														float *			pdKNNDistances,				// Out:	Distances of the K Nearest Neighbors in pdPosition.
														uint const		k,							// In:	Number of neighbors to consider.

														uint const		numB,						// In:	Size of group B.
														bool const		groupWithSelf				// In:	Are we testing this group with itself? (group A == group B)
														);
}

__inline__ __device__ int3 ComputeCellPos( volatile float3 const worldPosition )
{
	int3 cellPos;

	cellPos.x = (worldPosition.x + 0.5f * constWorldSizeV1.x) / constWorldStepV1.x;
	cellPos.y = (worldPosition.y + 0.5f * constWorldSizeV1.y) / constWorldStepV1.y;
	cellPos.z = (worldPosition.z + 0.5f * constWorldSizeV1.z) / constWorldStepV1.z;

	return cellPos;
}

__host__ void KNNBinningV1BindTexture( cudaArray * pdCudaArray )
{
	static cudaChannelFormatDesc const channelDesc = cudaCreateChannelDesc< uint >();

	texCellIndices.normalized = false;
	texCellIndices.filterMode = cudaFilterModePoint;
	// Wrap out of bounds coordinates.
	texCellIndices.addressMode[0] = cudaAddressModeWrap;
	texCellIndices.addressMode[1] = cudaAddressModeWrap;
	texCellIndices.addressMode[2] = cudaAddressModeWrap;

	CUDA_SAFE_CALL( cudaBindTextureToArray( texCellIndices, pdCudaArray, channelDesc ) );
}

__host__ void KNNBinningV1UnbindTexture( void )
{
	//CUDA_SAFE_CALL( cudaUnbindTexture( texCellIndicesNormalized ) );

	CUDA_SAFE_CALL( cudaUnbindTexture( texCellIndices ) );
}

__global__ void KNNBinningV1BuildDB(	float4 const*	pdPosition,				// In:	Positions of each agent.
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
	POSITION_SH( threadIdx.x ) = POSITION_F3( index );

	// Write the agent's cell index out to global memory.
	pdCellIndices[index] = CELL_INDEX( ComputeCellPos( POSITION_SH( threadIdx.x ) ) );

	// Write the agent's index out to global memory.
	pdAgentIndices[index] = index;
}

__host__ void KNNBinningV1ReorderDBBindTextures(	float4 const*	pdPosition,
												uint const		numAgents
												)
{
	static cudaChannelFormatDesc const float4ChannelDesc = cudaCreateChannelDesc< float4 >();

	CUDA_SAFE_CALL( cudaBindTexture( NULL, texPosition, pdPosition, float4ChannelDesc, numAgents * sizeof(float4) ) );
}


__host__ void KNNBinningV1ReorderDBUnbindTextures( void )
{
	CUDA_SAFE_CALL( cudaUnbindTexture( texPosition ) );
}

__global__ void KNNBinningV1ReorderDB(	uint const*		pdAgentIndices,		// In: (sorted) agent index.
										uint const*		pdCellIndices,		// In: (sorted) cell index agent is in.

										float4 *		pdPositionSorted,	// Out: Sorted agent positions.

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
	__shared__ float4 shPositionSorted[THREADSPERBLOCK];

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

	shPositionSorted[ threadIdx.x ] = tex1Dfetch( texPosition, iSortedIndex );

	// Write to global memory.
	pdPositionSorted[ index ] = shPositionSorted[ threadIdx.x ];
}

// Textures used by KNNBinningKernel.
texture< uint, cudaTextureType1D, cudaReadModeElementType>		texBCellStart;
texture< uint, cudaTextureType1D, cudaReadModeElementType>		texBCellEnd;
texture< uint, cudaTextureType1D, cudaReadModeElementType>		texBIndices;
texture< float4, cudaTextureType1D, cudaReadModeElementType>	texBPositionSorted;

__host__ void KNNBinningV1KernelBindTextures(	uint const*		pdBCellStart,
											uint const*		pdBCellEnd,
											uint const*		pdBIndices,
											float4 const*	pdBPositionSorted,
											uint const		numCells,
											uint const		numB
											)
{
	static cudaChannelFormatDesc const uintChannelDesc = cudaCreateChannelDesc< uint >();
	static cudaChannelFormatDesc const float4ChannelDesc = cudaCreateChannelDesc< float4 >();

	CUDA_SAFE_CALL( cudaBindTexture( NULL, texBCellStart, pdBCellStart, uintChannelDesc, numCells * sizeof(uint) ) );
	CUDA_SAFE_CALL( cudaBindTexture( NULL, texBCellEnd, pdBCellEnd, uintChannelDesc, numCells * sizeof(uint) ) );
	CUDA_SAFE_CALL( cudaBindTexture( NULL, texBIndices, pdBIndices, uintChannelDesc, numB * sizeof(uint) ) );
	CUDA_SAFE_CALL( cudaBindTexture( NULL, texBPositionSorted, pdBPositionSorted, float4ChannelDesc, numB * sizeof(float4) ) );
}

__host__ void KNNBinningV1KernelUnbindTextures( void )
{
	CUDA_SAFE_CALL( cudaUnbindTexture( texBCellStart ) );
	CUDA_SAFE_CALL( cudaUnbindTexture( texBCellEnd ) );
	CUDA_SAFE_CALL( cudaUnbindTexture( texBIndices ) );
	CUDA_SAFE_CALL( cudaUnbindTexture( texBPositionSorted ) );
}

__global__ void KNNBinningV1Kernel(	// Group A
									float4 const*	pdAPositionSorted,			// In:	Sorted group A positions.
									uint const*		pdAIndices,					// In:	Sorted group A indices
									uint const		numA,						// In:	Size of group A.

									// Cell neighbor info.
									int const		radius,						// In:	Search radius (in cells) to consider.

									// Output data.
									uint *			pdKNNIndices,				// Out:	Indices of K Nearest Neighbors in pdPosition.
									float *			pdKNNDistances,				// Out:	Distances of the K Nearest Neighbors in pdPosition.
									uint const		k,							// In:	Number of neighbors to consider.

									uint const		numB,						// In:	Size of group B.
									bool const		groupWithSelf				// In:	Are we testing this group with itself? (group A == group B)
									)
{
	// Index of this agent.
	int const AIndexSorted = (blockIdx.x * blockDim.x) + threadIdx.x;

	// Check bounds.
	if( AIndexSorted >= numA )
		return;

	__shared__ float3 shAPosition[THREADSPERBLOCK];

	// Shared memory for local priority queue computations.
	extern __shared__ uint shKNNIndices[];
	float * shKNNDistances = (float*)shKNNIndices + THREADSPERBLOCK * k;

	// Set all elements of shKNNDistances to FLT_MAX, shKNNIndices to UINT_MAX.
	for( uint i = 0; i < k; i++ )
	{
		shKNNIndices[(threadIdx.x * k) + i] = UINT_MAX;
		shKNNDistances[(threadIdx.x * k) + i] = FLT_MAX;
	}

	// Store this thread's index and cell index in registers.
	uint const		AIndex					= pdAIndices[ AIndexSorted ];

	// Coalesce read the positions.
	shAPosition[ threadIdx.x ] = make_float3( pdAPositionSorted[ AIndexSorted ] );

	int3 const cellPos = ComputeCellPos( shAPosition[ threadIdx.x ] );

	for( int dy = -radius; dy <= radius; dy++ )				// World height.
		for( int dz = -radius; dz <= radius; dz++ )			// World depth.
			for( int dx = -radius; dx <= radius; dx++ )		// World width.
			{
				int3 neighborPos = make_int3( cellPos.x + dx, cellPos.y + dy, cellPos.z + dz );
				uint const cellIndex = CELL_INDEX( neighborPos );

				// For each member of group B in the cell...
				for( uint BIndexSorted = tex1Dfetch( texBCellStart, cellIndex ) /*pdBCellStart[ cellIndex ]*/; BIndexSorted < tex1Dfetch( texBCellEnd, cellIndex ) /*pdBCellEnd[ cellIndex ]*/; BIndexSorted++ )
				{
					// Get the index of the other agent (unsorted).
					uint const BIndex = tex1Dfetch( texBIndices, BIndexSorted ) /*pdBIndices[ BIndexSorted ]*/;

					// Do not include self.
					if( groupWithSelf && AIndex == BIndex )
						continue;

					// Compute the distance between this thread'a A position and the B position at otherIndexSorted
					float const dist = float3_distance( shAPosition[threadIdx.x], make_float3( tex1Dfetch( texBPositionSorted, BIndexSorted ) ) );

					if( dist < shKNNDistances[(threadIdx.x * k) + (k - 1)] )	// Distance of the kth closest agent.
					{
						// Agent at index BIndex is the new (at least) kth closest. Set the distance and index in shared mem.
						shKNNDistances[(threadIdx.x * k) + (k - 1)] = dist;
						shKNNIndices[(threadIdx.x * k) + (k - 1)] = BIndex;

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
		pdKNNIndices[AIndex*k + i] = shKNNIndices[threadIdx.x*k + i];
		pdKNNDistances[AIndex*k + i] = shKNNDistances[threadIdx.x*k + i];
	}
	__syncthreads();
}