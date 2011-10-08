#include "KNNBinData.cuh"

#include "CUDAGlobals.cuh"

//#include "DebugUtils.h"

using namespace OpenSteer;

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

	__global__ void KNNBinningComputeCellNeighbors3D(	bin_cell const*	pdCells,			// In:	Cell data.
														uint *			pdCellNeighbors,	// Out:	Array of computed cell neighbors.
														size_t const	neighborsPerCell,	// In:	Number of neighbors per cell.
														uint const		radius,				// In:	Search radius.
														size_t const	numCells			// In:	Number of cells.
														);
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
	ComputeCellNeighbors( m_worldCells.y > 1 );
}

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

void KNNBinData::CreateCells( void )
{
	float3 const step =				make_float3(	m_worldSize.x / m_worldCells.x,		// width
													m_worldSize.y / m_worldCells.y,		// height
													m_worldSize.z / m_worldCells.z );	// depth

	float3 const stepNormalized =	make_float3(	step.x / m_worldSize.x,
													step.y / m_worldSize.y,
													step.z / m_worldSize.z
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
	size_t const numCells = m_worldCells.x * m_worldCells.y * m_worldCells.z;

	m_hvCells.resize( numCells );

	// Allocate host memory to temporarily store the 3D texture data.
	uint * phCellIndices = (uint*)malloc( numCells * sizeof(uint) );

	uint index = 0;

	for( size_t iHeight = 0; iHeight < m_worldCells.y; iHeight++ )		// height - texture z axis, world y axis
	{
		for( size_t iDepth = 0; iDepth < m_worldCells.z; iDepth++ )		// depth - texture y axis, world z axis
		{
			for( size_t iWidth = 0; iWidth < m_worldCells.x; iWidth++ )	// width - texture x axis, world x axis
			{
				// Make a bin_cell structure.
				bin_cell bc;

				//bc.iBinIndex = iBinIndex;
				bc.index = iWidth + (iDepth * m_worldCells.x) + (iHeight * m_worldCells.z * m_worldCells.x);

				// Set the offset value for the cell lookup texture.
				phCellIndices[index] = bc.index;

				// Set the minBounds of the cell.
				bc.minBound.x = iWidth * step.x - 0.5f * m_worldSize.x;
				bc.minBound.y = iHeight * step.y - 0.5f * m_worldSize.y;
				bc.minBound.z = iDepth * step.z - 0.5f * m_worldSize.z;

				// Set the position of the cell.
				bc.position.x = bc.minBound.x + 0.5f * step.x;
				bc.position.y = bc.minBound.y + 0.5f * step.y;
				bc.position.z = bc.minBound.z + 0.5f * step.z;

				// Set the maxBounds of the cell.
				bc.maxBound.x = bc.minBound.x + step.x;
				bc.maxBound.y = bc.minBound.y + step.y;
				bc.maxBound.z = bc.minBound.z + step.z;

				//m_hvCells.push_back( bc );
				m_hvCells[index] = bc;
				index++;
			}
		}
	}

	// Transfer the bin_cell structures to the device memory.
	m_dvCells = m_hvCells;

	// Prepare the bin_cell index lookup texture.
	cudaExtent const extent = make_cudaExtent( m_worldCells.x, m_worldCells.y, m_worldCells.z );
	cudaChannelFormatDesc const desc = cudaCreateChannelDesc< uint >();

	cudaPitchedPtr srcPtr = make_cudaPitchedPtr( (void*)phCellIndices, extent.width * sizeof(uint), extent.width, extent.height );

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
	CUDA_SAFE_CALL( cudaMemcpyToSymbol( "constWorldStep", &step, sizeof(float3) ) );
	CUDA_SAFE_CALL( cudaMemcpyToSymbol( "constWorldStepNormalized", &stepNormalized, sizeof(float3) ) );
	CUDA_SAFE_CALL( cudaMemcpyToSymbol( "constWorldCells", &m_worldCells, sizeof(uint3) ) );

	// Free host memory.
	free( phCellIndices );
}
