#include "KNNBinData.cuh"

#include "CUDAGlobals.cuh"

//#include "DebugUtils.h"

using namespace OpenSteer;

extern "C"
{
/*
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

	__global__ void KNNBinningComputeCellNeighbors3D(	bin_cell const*	pdCells,			// In:	Cell data.
														uint *			pdCellNeighbors,	// Out:	Array of computed cell neighbors.
														size_t const	neighborsPerCell,	// In:	Number of neighbors per cell.
														uint const		radius,				// In:	Search radius.
														size_t const	numCells			// In:	Number of cells.
														);
	*/
}

KNNBinData::KNNBinData( uint3 const& worldCells, float3 const& worldSize, uint const searchRadius )
:	m_worldCells( worldCells ),
	m_worldSize( worldSize ),
	m_nSearchRadius( searchRadius )
{
	m_nCells = m_worldCells.x * m_worldCells.y * m_worldCells.z;

	// Create the cells.
	CreateCells();
	
	// Compute the neighbors for the cells.
	ComputeCellNeighbors();
}

void KNNBinData::ComputeCellNeighbors( void )
{
	bool const b3D = m_worldCells.y > 1;

	m_nNeighborsPerCell		=  ipow( (m_nSearchRadius * 2 + 1), (b3D ? 3 : 2) );
	
	// Allocate temporary host-side storage.
	float3 * pNeighborOffsets = (float3*)malloc( m_nNeighborsPerCell * sizeof(float3) );

	if( b3D )
	{
		ComputeCellNeighbors3D( pNeighborOffsets );
		cudaMemcpyToSymbol( "constCellNeighborOffset3D", pNeighborOffsets, m_nNeighborsPerCell * sizeof(float3) );
	}
	else
	{
		ComputeCellNeighbors2D( pNeighborOffsets );
		cudaMemcpyToSymbol( "constCellNeighborOffset2D", pNeighborOffsets, m_nNeighborsPerCell * sizeof(float3) );
	}

	// Free the temporary memory.
	free( pNeighborOffsets );
}

#pragma warning ( push )
#pragma warning ( disable : 4018 )
void KNNBinData::ComputeCellNeighbors2D( float3 * pNeighborOffsets )
{
	// First offset is zero.
	pNeighborOffsets[0] = float3_zero();

	uint index = 1;

	// For increasing radius...
	for( int iCurrentRadius = 1; iCurrentRadius <= m_nSearchRadius; iCurrentRadius++ )
	{
		for( int dz = -iCurrentRadius; dz <= iCurrentRadius; dz++ )
		{
			for( int dx = -iCurrentRadius; dx <= iCurrentRadius; dx++ )
			{
				// Only do for the outside cells.
				if( dx == -iCurrentRadius || dx == iCurrentRadius || dz == -iCurrentRadius || dz == iCurrentRadius )
					pNeighborOffsets[ index++ ] = make_float3(	dx * m_worldStepNormalized.x,
																0.f,
																dz * m_worldStepNormalized.z
																);
			}
		}
	}
}

void KNNBinData::ComputeCellNeighbors3D( float3 * pNeighborOffsets )
{
	// First offset is zero.
	pNeighborOffsets[0] = float3_zero();

	uint index = 1;

	// For increasing radius...
	for( int iCurrentRadius = 1; iCurrentRadius <= m_nSearchRadius; iCurrentRadius++ )
	{
		for( int dy = -iCurrentRadius; dy <= iCurrentRadius; dy++ )			// World height.
		{
			for( int dz = -iCurrentRadius; dz <= iCurrentRadius; dz++ )		// World depth.
			{
				for( int dx = -iCurrentRadius; dx <= iCurrentRadius; dx++ )	// World width.
				{
					// Only do for the outside cells.
					if( dx == -iCurrentRadius || dx == iCurrentRadius || dz == -iCurrentRadius || dz == iCurrentRadius || dy == -iCurrentRadius || dy == iCurrentRadius )
						pNeighborOffsets[ index++ ] = make_float3(	dx * m_worldStepNormalized.x,
																	dy * m_worldStepNormalized.y,
																	dz * m_worldStepNormalized.z
																	);
				}
			}
		}
	}
}
#pragma warning ( pop )

/*
void KNNBinData::ComputeCellNeighbors( bool b3D )
{
	dim3 grid = gridDim();
	dim3 block = blockDim();

	m_nNeighborsPerCell		=  ipow( (m_nSearchRadius * 2 + 1), (b3D ? 3 : 2) );
	size_t const shMemSize	= sizeof(uint) * KNN_THREADSPERBLOCK * m_nNeighborsPerCell;

	// Allocate enough device memory.
	m_dvCellNeighbors.resize( m_nCells * m_nNeighborsPerCell );

	// Bind the texture.
	KNNBinningCUDABindTexture( pdCellIndexArray() );

	if( b3D )
	{
		KNNBinningComputeCellNeighbors3D<<< grid, block, shMemSize >>>( pdCells(), pdCellNeighbors(), m_nNeighborsPerCell, m_nSearchRadius, m_nCells );
	}
	else
	{
		KNNBinningComputeCellNeighbors2D<<< grid, block, shMemSize >>>( pdCells(), pdCellNeighbors(), m_nNeighborsPerCell, m_nSearchRadius, m_nCells );
	}
	cutilCheckMsg( "KNNBinningComputeCellNeighbors failed." );
	CUDA_SAFE_CALL( cudaThreadSynchronize() );

	// Unbind the texture.
	KNNBinningCUDAUnbindTexture();
}
*/

void KNNBinData::CreateCells( void )
{
	m_worldStep				= make_float3(	m_worldSize.x / m_worldCells.x,		// width
											m_worldSize.y / m_worldCells.y,		// depth
											m_worldSize.z / m_worldCells.z );	// height

	m_worldStepNormalized	= make_float3(	m_worldStep.x / m_worldSize.x,
											m_worldStep.y / m_worldSize.y,
											m_worldStep.z / m_worldSize.z
											);

/*
Texture addressing in CUDA operates as follows.
   z|
	|    y/
	|    /
	|   /
	|  /
	| /
	|/_________x
*/
	//m_hvCells.resize( numCells );
	m_hvPositions.reserve( m_nCells );
	m_hvMinBounds.reserve( m_nCells );
	m_hvMaxBounds.reserve( m_nCells );

	// Allocate host memory to temporarily store the 3D texture data.
	uint * phCellIndices = (uint*)malloc( m_nCells * sizeof(uint) );

	uint index = 0;

	for( size_t iHeight = 0; iHeight < m_worldCells.y; iHeight++ )		// height - texture z axis, world y axis
	{
		for( size_t iDepth = 0; iDepth < m_worldCells.z; iDepth++ )		// depth - texture y axis, world z axis
		{
			for( size_t iWidth = 0; iWidth < m_worldCells.x; iWidth++ )	// width - texture x axis, world x axis
			{
				// Set the offset value for the cell lookup texture.
				phCellIndices[index++] = iWidth + (iDepth * m_worldCells.x) + (iHeight * m_worldCells.z * m_worldCells.x);

				float3 position, minBound, maxBound;

				// Set the minBounds of the cell.
				minBound.x = iWidth * m_worldStep.x - 0.5f * m_worldSize.x;
				minBound.y = iHeight * m_worldStep.y - 0.5f * m_worldSize.y;
				minBound.z = iDepth * m_worldStep.z - 0.5f * m_worldSize.z;

				// Set the position of the cell.
				position.x = minBound.x + 0.5f * m_worldStep.x;
				position.y = minBound.y + 0.5f * m_worldStep.y;
				position.z = minBound.z + 0.5f * m_worldStep.z;

				// Set the maxBounds of the cell.
				maxBound.x = minBound.x + m_worldStep.x;
				maxBound.y = minBound.y + m_worldStep.y;
				maxBound.z = minBound.z + m_worldStep.z;

				m_hvPositions.push_back( position );
				m_hvMinBounds.push_back( minBound );
				m_hvMaxBounds.push_back( maxBound );
			}
		}
	}

	// Transfer the cell position and bounds data to device memory.
	m_dvPositions = m_hvPositions;
	m_dvMinBounds = m_hvMinBounds;
	m_dvMaxBounds = m_hvMaxBounds;

	// Prepare the bin_cell index lookup texture.
	cudaExtent const extent = make_cudaExtent( m_worldCells.x, m_worldCells.y, m_worldCells.z );
	cudaChannelFormatDesc const desc = cudaCreateChannelDesc< uint >();

	cudaPitchedPtr srcPtr = make_cudaPitchedPtr( (void*)phCellIndices, m_worldCells.x * sizeof(uint), m_worldCells.x, m_worldCells.y );

	// Allocate m_pdCellIndexArray.
	CUDA_SAFE_CALL( cudaMalloc3DArray( &m_pdCellIndexArray, &desc, extent, cudaArrayDefault ) );

	// Copy data to 3D array.
	cudaMemcpy3DParms copyParms = {0};
	copyParms.srcPtr = srcPtr;
	copyParms.dstArray = m_pdCellIndexArray;
	copyParms.extent = extent;
	copyParms.kind = cudaMemcpyHostToDevice;
	CUDA_SAFE_CALL( cudaMemcpy3D( &copyParms ) );

	// Copy the m_worldSize and m_worldCells values to constant memory.
	CUDA_SAFE_CALL( cudaMemcpyToSymbol( "constWorldSize", &m_worldSize, sizeof(float3) ) );
	CUDA_SAFE_CALL( cudaMemcpyToSymbol( "constWorldStep", &m_worldStep, sizeof(float3) ) );
	CUDA_SAFE_CALL( cudaMemcpyToSymbol( "constWorldStepNormalized", &m_worldStepNormalized, sizeof(float3) ) );
	CUDA_SAFE_CALL( cudaMemcpyToSymbol( "constWorldCells", &m_worldCells, sizeof(uint3) ) );

	// Free host memory.
	free( phCellIndices );
}
