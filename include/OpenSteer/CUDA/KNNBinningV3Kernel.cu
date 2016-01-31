#include "KNNBinningV3.cuh"

#include "../VectorUtils.cuh"

#include "CUDAKernelGlobals.cuh"

using namespace OpenSteer;

#define SEEDING

// Texture references.
texture< uint, cudaTextureType3D, cudaReadModeElementType >		texCellIndices;
texture< uint, cudaTextureType3D, cudaReadModeElementType >		texCellIndicesNormalized;
texture< float4, cudaTextureType1D, cudaReadModeElementType >	texPosition;

__constant__ float3		constWorldSizeV3;
__constant__ float3		constWorldStepV3;
__constant__ float3		constWorldStepNormalizedV3;
__constant__ uint3		constWorldCellsV3;

// Fetch the cell index from texCellIndicesNormalized at a given world {x,y,z} position.
#define CELL_INDEX_NORMALIZED( pos )	( tex3D( texCellIndicesNormalized, pos.x, pos.z, pos.y ) )	// <--- for boids.
//#define CELL_INDEX_NORMALIZED( pos )	( tex3D( texCellIndicesNormalized, pos.x, pos.y, pos.z ) )	// <--- for choke point.

// Fetch the cell index from texCellIndices at a given texel (x,y,z) coordinate.
//#define CELL_INDEX( x, y, z )			( tex3D( texCellIndices, x, y, z ) )
#define CELL_INDEX( pos )	( tex3D( texCellIndices, pos.x, pos.z, pos.y ) )	// <--- for boids.
//#define CELL_INDEX( pos )	( tex3D( texCellIndices, pos.x, pos.y, pos.z ) )	// <--- for choke point.

// Kernel declarations.
extern "C"
{
	// Bind the textures to the input cudaArray.
	__host__ void KNNBinningV3BindTexture(		cudaArray * pCudaArray );
	// Unbind the textures.
	__host__ void KNNBinningV3UnbindTexture(		void );

	__host__ void KNNBinningV3KernelBindTextures(		uint const*		pdBCellStart,
														uint const*		pdBCellEnd,
														uint const*		pdBIndices,
														float4 const*	pdBPosition,
														float4 const*	pdBPositionSorted,
														uint const		numCells,
														uint const		numB,
														uint const		neighborsPerCell,
														uint const*		pdCellNeighbours
														);

	__host__ void KNNBinningV3KernelUnbindTextures(	void );
	__host__ void KNNBinningV3ReorderDBBindTextures(	float4 const*	pdPosition,
													uint const		numAgents );
	__host__ void KNNBinningV3ReorderDBUnbindTextures( void );

	// Use to precompute the neighbors of each cell once per decomposition.
	__global__ void KNNBinningV3ComputeCellNeighbors2D(	bin_cell const*	pdCells,			// In:	Cell data.
														uint *			pdCellNeighbors,	// Out:	Array of computed cell neighbors.
														size_t const	neighborsPerCell,	// In:	Number of neighbors per cell.
														uint const		radius,				// In:	Search radius.
														size_t const	numCells			// In:	Number of cells.
														);

	__global__ void KNNBinningV3ComputeCellNeighbors3D(	bin_cell const*	pdCells,			// In:	Cell data.
														uint *			pdCellNeighbors,	// Out:	Array of computed cell neighbors.
														size_t const	neighborsPerCell,	// In:	Number of neighbors per cell.
														uint const		radius,				// In:	Search radius.
														size_t const	numCells			// In:	Number of cells.
														);

	// Kernel to set initial bin indices of vehicles in the simulation.
	__global__ void KNNBinningV3BuildDB(					float4 const*	pdPosition,				// In:	Positions of each agent.
														size_t *		pdAgentIndices,			// Out:	Indices of each agent.
														size_t *		pdCellIndices,			// Out:	Indices of the cell each agent is in.
														size_t const	numAgents
														);

	// Reorder the positions on pdCellIndices, and compute the cell start and end indices.
	__global__ void KNNBinningV3ReorderDB(				uint const*		pdAgentIndices,		// In: (sorted) agent index.
														uint const*		pdCellIndices,		// In: (sorted) cell index agent is in.

														float4 *		pdPositionSorted,	// Out: Sorted agent positions.

														uint *			pdCellStart,		// Out: Start index of this cell in pdCellIndices.
														uint *			pdCellEnd,			// Out: End index of this cell in pdCellIndices.

														size_t const	numAgents
														);

	__global__ void KNNBinningV3Kernel(					// Group A
														float4 const*	pdAPositionSorted,			// In:	Sorted group A positions.

														uint const*		pdAIndices,					// In:	Sorted group A indices
														uint const*		pdACellIndices,				// In:	Sorted group A cell indices.

														// Cell neighbor info.
														uint const		neighborsPerCell,			// In:	Number of neighbors per cell in the pdCellNeighbors array.
														uint const		radius,						// In:	Search radius (in cells) to consider.
														uint const		numCells,
														uint const*		pdCellNeighbours,


														// Output data.
														uint *			pdKNNIndices,				// Out:	Indices of K Nearest Neighbors in pdPosition.
														float *			pdKNNDistances,				// Out:	Distances of the K Nearest Neighbors in pdPosition.

														uint const		k,							// In:	Number of neighbors to consider.
														uint const		numA,						// In:	Size of group A.
														uint const		numB,						// In:	Size of group B.
														bool const		groupWithSelf,				// In:	Are we testing this group with itself? (group A == group B)
														bool const		bSeed
														);
}

__inline__ __device__ int3 ComputeCellPos( volatile float3 const worldPosition )
{
	int3 cellPos;

	cellPos.x = (worldPosition.x + 0.5f * constWorldSizeV3.x) / constWorldStepV3.x;
	cellPos.y = (worldPosition.y + 0.5f * constWorldSizeV3.y) / constWorldStepV3.y;
	cellPos.z = (worldPosition.z + 0.5f * constWorldSizeV3.z) / constWorldStepV3.z;

	return cellPos;
}

__global__ void KNNBinningV3ComputeCellNeighbors3D(	bin_cell const*	pdCells,			// In:	Cell data.
													uint *			pdCellNeighbors,	// Out:	Array of computed cell neighbors.
													size_t const	neighborsPerCell,	// In:	Number of neighbors per cell.
													uint const		radius,				// In:	Search radius.
													size_t const	numCells			// In:	Number of cells.
													)
{
	int const index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if( index >= numCells )
		return;

	__shared__ float3 shPosition[KNN_THREADSPERBLOCK];
	extern __shared__ uint shNeighboringCells[];
	
	// Read the position of this thread's cell to shared memory.
	shPosition[ threadIdx.x ] = pdCells[index].position;

	// Normalize the positions.
	POSITION_SH( threadIdx.x ).x = (POSITION_SH( threadIdx.x ).x + 0.5f * constWorldSizeV3.x) / constWorldSizeV3.x;
	POSITION_SH( threadIdx.x ).y = (POSITION_SH( threadIdx.x ).y + 0.5f * constWorldSizeV3.y) / constWorldSizeV3.y;
	POSITION_SH( threadIdx.x ).z = (POSITION_SH( threadIdx.x ).z + 0.5f * constWorldSizeV3.z) / constWorldSizeV3.z;

	// Get the first cell index (radius 0).
	shNeighboringCells[ threadIdx.x * neighborsPerCell ] = CELL_INDEX_NORMALIZED( POSITION_SH( threadIdx.x ) );

	__syncthreads();

	int i = 1;
	// Compute the start offset into shNeighboringCells for this radius.
	int offset = threadIdx.x * neighborsPerCell;

	// For increasing radius...
	for( int iCurrentRadius = 1; iCurrentRadius <= radius; iCurrentRadius++ )
	{
		for( int dy = -iCurrentRadius; dy <= iCurrentRadius; dy++ )			// World height.
		{
			for( int dz = -iCurrentRadius; dz <= iCurrentRadius; dz++ )		// World depth.
			{
				for( int dx = -iCurrentRadius; dx <= iCurrentRadius; dx++ )	// World width.
				{
					// Only do for the outside cells.
					if(	dz == -iCurrentRadius || dz == iCurrentRadius ||
						dx == -iCurrentRadius || dx == iCurrentRadius ||
						dy == -iCurrentRadius || dy == iCurrentRadius
						)
					{
						float3 queryPosition = make_float3(	POSITION_SH( threadIdx.x ).x + dx * constWorldStepNormalizedV3.x,
															POSITION_SH( threadIdx.x ).y + dy * constWorldStepNormalizedV3.y,
															POSITION_SH( threadIdx.x ).z + dz * constWorldStepNormalizedV3.z
															);

						uint cellIndex = CELL_INDEX_NORMALIZED( queryPosition );

						// Do not add duplicate cells.
						for( int iDup = 0; iDup < i; iDup++ )
						{
							if( shNeighboringCells[offset+iDup] == cellIndex )
							{
								cellIndex = UINT_MAX;
								break;
							}
						}

						shNeighboringCells[offset + i++] = cellIndex;
					}
				}
			}
		}
	}

	__syncthreads();
	for( int i = 0; i < neighborsPerCell; i++ )
	{
		//pdCellNeighbors[ index * neighborsPerCell + i ] = shNeighboringCells[ offset + i ];
		pdCellNeighbors[ numCells * i + index ] = shNeighboringCells[ offset + i ];
	}
}

__global__ void KNNBinningV3ComputeCellNeighbors2D(	bin_cell const*	pdCells,			// In:	Cell data.
													uint *			pdCellNeighbors,	// Out:	Array of computed cell neighbors.
													size_t const	neighborsPerCell,	// In:	Number of neighbors per cell.
													uint const		radius,				// In:	Search radius.
													size_t const	numCells			// In:	Number of cells.
													)
{
	int const index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if( index >= numCells )
		return;

	__shared__ float3 shPosition[KNN_THREADSPERBLOCK];
	extern __shared__ uint shNeighboringCells[];
	
	// Read the position of this thread's cell to shared memory.
	shPosition[ threadIdx.x ] = pdCells[index].position;

	// Normalize the positions.
	POSITION_SH( threadIdx.x ).x = (POSITION_SH( threadIdx.x ).x + 0.5f * constWorldSizeV3.x) / constWorldSizeV3.x;
	POSITION_SH( threadIdx.x ).y = (POSITION_SH( threadIdx.x ).y + 0.5f * constWorldSizeV3.y) / constWorldSizeV3.y;
	POSITION_SH( threadIdx.x ).z = (POSITION_SH( threadIdx.x ).z + 0.5f * constWorldSizeV3.z) / constWorldSizeV3.z;

	// Get the first cell index (radius 0).
	shNeighboringCells[ threadIdx.x * neighborsPerCell ] = CELL_INDEX_NORMALIZED( POSITION_SH( threadIdx.x ) );

	__syncthreads();

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
					float3 queryPosition = make_float3(	POSITION_SH( threadIdx.x ).x + dx * constWorldStepNormalizedV3.x,
														POSITION_SH( threadIdx.x ).y,
														POSITION_SH( threadIdx.x ).z + dz * constWorldStepNormalizedV3.z
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
		//pdCellNeighbors[ index * neighborsPerCell + i ] = shNeighboringCells[ offset + i ];
		pdCellNeighbors[ numCells * i + index ] = shNeighboringCells[ offset + i ];
	}
}

__host__ void KNNBinningV3BindTexture( cudaArray * pdCudaArray )
{
	static cudaChannelFormatDesc const channelDesc = cudaCreateChannelDesc< uint >();

	texCellIndices.normalized = false;
	texCellIndices.filterMode = cudaFilterModePoint;
	// Clamp out of bounds coordinates to the edge of the texture.
	texCellIndices.addressMode[0] = cudaAddressModeWrap;
	texCellIndices.addressMode[1] = cudaAddressModeWrap;
	texCellIndices.addressMode[2] = cudaAddressModeWrap;

	CUDA_SAFE_CALL( cudaBindTextureToArray( texCellIndices, pdCudaArray, channelDesc ) );

	texCellIndicesNormalized.normalized = true;
	texCellIndicesNormalized.filterMode = cudaFilterModePoint;
	// Clamp out of bounds coordinates to the edge of the texture.
	texCellIndicesNormalized.addressMode[0] = cudaAddressModeWrap;
	texCellIndicesNormalized.addressMode[1] = cudaAddressModeWrap;
	texCellIndicesNormalized.addressMode[2] = cudaAddressModeWrap;

	CUDA_SAFE_CALL( cudaBindTextureToArray( texCellIndicesNormalized, pdCudaArray, channelDesc ) );
}

__host__ void KNNBinningV3UnbindTexture( void )
{
	CUDA_SAFE_CALL( cudaUnbindTexture( texCellIndices ) );
	CUDA_SAFE_CALL( cudaUnbindTexture( texCellIndicesNormalized ) );
}

__global__ void KNNBinningV3BuildDB(	float4 const*	pdPosition,				// In:	Positions of each agent.
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
	//__shared__ float3 shPosition[THREADSPERBLOCK];
	//POSITION_SH( threadIdx.x ) = POSITION_F3( index );

	// Normalize the positions.
	//POSITION_SH( threadIdx.x ).x = (POSITION_SH( threadIdx.x ).x + 0.5f * constWorldSizeV3.x) / constWorldSizeV3.x;
	//POSITION_SH( threadIdx.x ).y = (POSITION_SH( threadIdx.x ).y + 0.5f * constWorldSizeV3.y) / constWorldSizeV3.y;
	//POSITION_SH( threadIdx.x ).z = (POSITION_SH( threadIdx.x ).z + 0.5f * constWorldSizeV3.z) / constWorldSizeV3.z;
	
	// Write the agent's cell index out to global memory.
	//pdCellIndices[index] = CELL_INDEX_NORMALIZED( POSITION_SH( threadIdx.x ) );
	pdCellIndices[ index ] = CELL_INDEX( ComputeCellPos( POSITION_F3( index ) ) );

	// Write the agent's index out to global memory.
	pdAgentIndices[ index ] = index;
}


__host__ void KNNBinningV3ReorderDBBindTextures(	float4 const*	pdPosition,
												uint const		numAgents
												)
{
	static cudaChannelFormatDesc const float4ChannelDesc = cudaCreateChannelDesc< float4 >();

	CUDA_SAFE_CALL( cudaBindTexture( NULL, texPosition, pdPosition, float4ChannelDesc, numAgents * sizeof(float4) ) );
}


__host__ void KNNBinningV3ReorderDBUnbindTextures( void )
{
	CUDA_SAFE_CALL( cudaUnbindTexture( texPosition ) );
}

__global__ void KNNBinningV3ReorderDB(	uint const*		pdAgentIndices,		// In: (sorted) agent index.
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
	//__shared__ float4 shPositionSorted[THREADSPERBLOCK];

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

	//shPositionSorted[ threadIdx.x ] = tex1Dfetch( texPosition, iSortedIndex );

	//// Write to global memory.
	//pdPositionSorted[ index ] = shPositionSorted[ threadIdx.x ];

	// Write the sorted position to global memory.
	pdPositionSorted[ index ] = tex1Dfetch( texPosition, iSortedIndex );
}

// Textures used by KNNBinningKernel.
texture< uint, cudaTextureType1D, cudaReadModeElementType>		texBCellStart;
texture< uint, cudaTextureType1D, cudaReadModeElementType>		texBCellEnd;
texture< uint, cudaTextureType1D, cudaReadModeElementType>		texBIndices;
texture< float4, cudaTextureType1D, cudaReadModeElementType>	texBPosition;
texture< float4, cudaTextureType1D, cudaReadModeElementType>	texBPositionSorted;
texture< uint, cudaTextureType1D, cudaReadModeElementType>		texCellNeighbours;

__host__ void KNNBinningV3KernelBindTextures(	uint const*		pdBCellStart,
												uint const*		pdBCellEnd,
												uint const*		pdBIndices,
												float4 const*	pdBPosition,
												float4 const*	pdBPositionSorted,
												uint const		numCells,
												uint const		numB,
												uint const		neighborsPerCell,
												uint const*		pdCellNeighbours
												)
{
	static cudaChannelFormatDesc const uintChannelDesc = cudaCreateChannelDesc< uint >();
	static cudaChannelFormatDesc const float4ChannelDesc = cudaCreateChannelDesc< float4 >();

	CUDA_SAFE_CALL( cudaBindTexture( NULL, texBCellStart, pdBCellStart, uintChannelDesc, numCells * sizeof(uint) ) );
	CUDA_SAFE_CALL( cudaBindTexture( NULL, texBCellEnd, pdBCellEnd, uintChannelDesc, numCells * sizeof(uint) ) );
	CUDA_SAFE_CALL( cudaBindTexture( NULL, texBIndices, pdBIndices, uintChannelDesc, numB * sizeof(uint) ) );
	CUDA_SAFE_CALL( cudaBindTexture( NULL, texBPosition, pdBPosition, float4ChannelDesc, numB * sizeof(float4) ) );
	CUDA_SAFE_CALL( cudaBindTexture( NULL, texBPositionSorted, pdBPositionSorted, float4ChannelDesc, numB * sizeof(float4) ) );
	CUDA_SAFE_CALL( cudaBindTexture( NULL, texCellNeighbours, pdCellNeighbours, uintChannelDesc, numCells * neighborsPerCell * sizeof(uint) ) );
}

__host__ void KNNBinningV3KernelUnbindTextures( void )
{
	CUDA_SAFE_CALL( cudaUnbindTexture( texBCellStart ) );
	CUDA_SAFE_CALL( cudaUnbindTexture( texBCellEnd ) );
	CUDA_SAFE_CALL( cudaUnbindTexture( texBIndices ) );
	CUDA_SAFE_CALL( cudaUnbindTexture( texBPosition ) );
	CUDA_SAFE_CALL( cudaUnbindTexture( texBPositionSorted ) );
	CUDA_SAFE_CALL( cudaUnbindTexture( texCellNeighbours ) );
}

#if defined SEEDING
__global__ void KNNBinningV3Kernel(	// Group A
									float4 const*	pdAPositionSorted,			// In:	Sorted group A positions.

									uint const*		pdAIndices,					// In:	Sorted group A indices
									uint const*		pdACellIndices,				// In:	Sorted group A cell indices.

									// Cell neighbor info.
									uint const		neighborsPerCell,			// In:	Number of neighbors per cell in the pdCellNeighbors array.
									uint const		radius,						// In:	Search radius (in cells) to consider.
									uint const		numCells,
									uint const*		pdCellNeighbours,

									// Output data.
									uint *			pdKNNIndices,				// Out:	Indices of K Nearest Neighbors in pdPosition.
									float *			pdKNNDistances,				// Out:	Distances of the K Nearest Neighbors in pdPosition.

									uint const		k,							// In:	Number of neighbors to consider.
									uint const		numA,						// In:	Size of group A.
									uint const		numB,						// In:	Size of group B.
									bool const		groupWithSelf,				// In:	Are we testing this group with itself? (group A == group B)
									bool const		bSeed
									)
{
	// Index of this agent.
	int const AIndexSorted = (blockIdx.x * blockDim.x) + threadIdx.x;

	// Check bounds.
	if( AIndexSorted >= numA )
		return;

	// Store this thread's index and cell index in registers.
	uint const		AIndex					= pdAIndices[ AIndexSorted ];
	uint const		cellIndex				= pdACellIndices[ AIndexSorted ];

	__shared__ float3 shAPosition[THREADSPERBLOCK];

	// Coalesce read the positions.
	shAPosition[ threadIdx.x ] = make_float3( pdAPositionSorted[ AIndexSorted ] );

	// Shared memory for local priority queue computations.
	extern __shared__ uint shKNNIndices[];
	float * shKNNDistances = (float*)shKNNIndices + THREADSPERBLOCK * k;

	if( bSeed )
	{
		// Load the 
		for( uint i = 0; i < k; i++ )
		{
			uint const knnInd = pdKNNIndices[ numA * i + AIndex ];
			shKNNIndices[ THREADSPERBLOCK * i + threadIdx.x ] = knnInd;
			shKNNDistances[ THREADSPERBLOCK * i + threadIdx.x ] = (knnInd < numB)	? float3_distance( shAPosition[ threadIdx.x ], make_float3( tex1Dfetch( texBPosition, knnInd ) ) )
																					: FLT_MAX;
		}

		// Re-sort using bubble sort.
		bool sorted;
		do
		{
			sorted = true;

			for( uint i = 1; i < k; i++ )
			{
				if( shKNNDistances[ THREADSPERBLOCK * (i-1) + threadIdx.x ] > shKNNDistances[ THREADSPERBLOCK * i + threadIdx.x ] )
				{
					sorted = false;

					swap( shKNNDistances[ THREADSPERBLOCK * (i-1) + threadIdx.x ], shKNNDistances[ THREADSPERBLOCK * i + threadIdx.x ] );
					swap( shKNNIndices[ THREADSPERBLOCK * (i-1) + threadIdx.x ], shKNNIndices[ THREADSPERBLOCK * i + threadIdx.x ] );
				}
			}
		} while( ! sorted );
	}
	else
	{
		// Set all elements of shKNNDistances to FLT_MAX, shKNNIndices to UINT_MAX.
		for( uint i = 0; i < k; i++ )
		{
			shKNNIndices[ THREADSPERBLOCK * i + threadIdx.x ] = UINT_MAX;
			shKNNDistances[ THREADSPERBLOCK * i + threadIdx.x ] = FLT_MAX;
		}
	}

	__syncthreads();

	// For each of the neighbors of the current cell...
	for( int iCellNeighbor = 0; iCellNeighbor < neighborsPerCell; iCellNeighbor++ )
	{
		// Get the index of this neighbor.
		//uint neighboringCellIndex = pdCellNeighbours[ numCells * iCellNeighbor + cellIndex ];
		uint neighboringCellIndex = tex1Dfetch( texCellNeighbours, numCells * iCellNeighbor + cellIndex );

		if( neighboringCellIndex == UINT_MAX )	// There is no neighboring cell in this position.
			continue;

		uint const BCellEnd = tex1Dfetch( texBCellEnd, neighboringCellIndex );

		// For each member of group B in the cell...
		for( uint BIndexSorted = tex1Dfetch( texBCellStart, neighboringCellIndex ); BIndexSorted < BCellEnd; BIndexSorted++ )
		{
			// Get the index of the other agent (unsorted).
			uint const BIndex = tex1Dfetch( texBIndices, BIndexSorted );
			
			// Do not include self.
			if( groupWithSelf && AIndex == BIndex )
				continue;

			// Compute the distance between this thread'a A position and the B position at otherIndexSorted
			float const dist = float3_distance( shAPosition[threadIdx.x], make_float3( tex1Dfetch( texBPositionSorted, BIndexSorted ) ) );

			if( dist < shKNNDistances[ THREADSPERBLOCK * (k-1) + threadIdx.x ] )	// Distance of the kth closest agent.
			{
				bool bSeeded = false;

				for( uint i = 0; i < k; i++ )
				{
					if( shKNNIndices[ THREADSPERBLOCK * i + threadIdx.x ] == BIndex )
						bSeeded = true;
				}

				if( ! bSeeded )
				{
					// Agent at index BIndex is the new (at least) kth closest. Set the distance and index in shared mem.
					shKNNDistances[ THREADSPERBLOCK * (k-1) + threadIdx.x ] = dist;
					shKNNIndices[ THREADSPERBLOCK * (k-1) + threadIdx.x ] = BIndex;

					// Bubble the values up...
					for( int slot = k - 2; slot >= 0; slot-- )
					{
						if( shKNNDistances[ THREADSPERBLOCK * slot + threadIdx.x ] > shKNNDistances[ THREADSPERBLOCK * (slot+1) + threadIdx.x ] )
						{
							swap( shKNNDistances[ THREADSPERBLOCK * slot + threadIdx.x ], shKNNDistances[ THREADSPERBLOCK * (slot+1) + threadIdx.x ] );
							swap( shKNNIndices[ THREADSPERBLOCK * slot + threadIdx.x ], shKNNIndices[ THREADSPERBLOCK * (slot+1) + threadIdx.x ] );
						}
						else
							break;
					}
				}
			}
		}
	}

	__syncthreads();

	// Write the shKNNIndices and shKNNDistances values out to global memory.
	for( uint i = 0; i < k; i++ )
	{
		pdKNNIndices[ numA * i + AIndex ] = shKNNIndices[ THREADSPERBLOCK * i + threadIdx.x ];
		pdKNNDistances[ numA * i + AIndex ] = shKNNDistances[ THREADSPERBLOCK * i + threadIdx.x ];
	}
}
#else
__global__ void KNNBinningV3Kernel(	// Group A
									float4 const*	pdAPositionSorted,			// In:	Sorted group A positions.

									uint const*		pdAIndices,					// In:	Sorted group A indices
									uint const*		pdACellIndices,				// In:	Sorted group A cell indices.

									// Cell neighbor info.
									uint const		neighborsPerCell,			// In:	Number of neighbors per cell in the pdCellNeighbors array.
									uint const		radius,						// In:	Search radius (in cells) to consider.
									uint const		numCells,
									uint const*		pdCellNeighbours,


									// Output data.
									uint *			pdKNNIndices,				// Out:	Indices of K Nearest Neighbors in pdPosition.
									float *			pdKNNDistances,				// Out:	Distances of the K Nearest Neighbors in pdPosition.

									uint const		k,							// In:	Number of neighbors to consider.
									uint const		numA,						// In:	Size of group A.
									uint const		numB,						// In:	Size of group B.
									bool const		groupWithSelf,				// In:	Are we testing this group with itself? (group A == group B)
									bool const		bSeed
									)
{
	// Index of this agent.
	int const AIndexSorted = (blockIdx.x * blockDim.x) + threadIdx.x;

	// Check bounds.
	if( AIndexSorted >= numA )
		return;

	// Store this thread's index and cell index in registers.
	uint const		AIndex					= pdAIndices[ AIndexSorted ];
	uint const		cellIndex				= pdACellIndices[ AIndexSorted ];

	__shared__ float3 shAPosition[THREADSPERBLOCK];

	// Coalesce read the positions.
	shAPosition[ threadIdx.x ] = make_float3( pdAPositionSorted[ AIndexSorted ] );

	// Shared memory for local priority queue computations.
	extern __shared__ uint shKNNIndices[];
	float * shKNNDistances = (float*)shKNNIndices + THREADSPERBLOCK * k;

	// Set all elements of shKNNDistances to FLT_MAX, shKNNIndices to UINT_MAX.
	for( uint i = 0; i < k; i++ )
	{
		shKNNIndices[ THREADSPERBLOCK * i + threadIdx.x ] = UINT_MAX;
		shKNNDistances[ THREADSPERBLOCK * i + threadIdx.x ] = FLT_MAX;
	}

	// For each of the neighbors of the current cell...
	for( int iCellNeighbor = 0; iCellNeighbor < neighborsPerCell; iCellNeighbor++ )
	{
		// Get the index of this neighbor.
		uint neighboringCellIndex = pdCellNeighbours[ cellIndex + iCellNeighbor * numCells ];

		if( neighboringCellIndex == UINT_MAX )	// There is no neighboring cell in this position.
			continue;

		uint const BCellEnd = tex1Dfetch( texBCellEnd, neighboringCellIndex );

		// For each member of group B in the cell...
		for( uint BIndexSorted = tex1Dfetch( texBCellStart, neighboringCellIndex ); BIndexSorted < BCellEnd; BIndexSorted++ )
		{
			// Get the index of the other agent (unsorted).
			uint const BIndex = tex1Dfetch( texBIndices, BIndexSorted );
			
			// Do not include self.
			if( groupWithSelf && AIndex == BIndex )
				continue;

			// Compute the distance between this thread'a A position and the B position at otherIndexSorted
			float const dist = float3_distance( shAPosition[threadIdx.x], make_float3( tex1Dfetch( texBPositionSorted, BIndexSorted ) ) );

			if( dist < shKNNDistances[ THREADSPERBLOCK * (k-1) + threadIdx.x ] )	// Distance of the kth closest agent.
			{
				// Agent at index BIndex is the new (at least) kth closest. Set the distance and index in shared mem.
				shKNNDistances[ THREADSPERBLOCK * (k-1) + threadIdx.x ] = dist;
				shKNNIndices[ THREADSPERBLOCK * (k-1) + threadIdx.x ] = BIndex;

				// Bubble the values up...
				for( int slot = k - 2; slot >= 0; slot-- )
				{
					if( shKNNDistances[ THREADSPERBLOCK * slot + threadIdx.x ] > shKNNDistances[ THREADSPERBLOCK * (slot+1) + threadIdx.x ] )
					{
						swap( shKNNDistances[ THREADSPERBLOCK * slot + threadIdx.x ], shKNNDistances[ THREADSPERBLOCK * (slot+1) + threadIdx.x ] );
						swap( shKNNIndices[ THREADSPERBLOCK * slot + threadIdx.x ], shKNNIndices[ THREADSPERBLOCK * (slot+1) + threadIdx.x ] );
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
		pdKNNIndices[ numA * i + AIndex ] = shKNNIndices[ THREADSPERBLOCK * i + threadIdx.x ];
		pdKNNDistances[ numA * i + AIndex ] = shKNNDistances[ THREADSPERBLOCK * i + threadIdx.x ];
	}
	__syncthreads();
}
#endif
