#include "KNNBinDataV1.cuh"

#include "CUDAGlobals.cuh"

//#include "DebugUtils.h"

using namespace OpenSteer;

extern "C"
{
	// Bind the textures to the input cudaArray.
	__host__ void KNNBinningV1BindTexture( cudaArray * pCudaArray );
	// Unbind the textures.
	__host__ void KNNBinningV1UnbindTexture( void );
}

KNNBinDataV1::KNNBinDataV1( uint3 const& worldCells, float3 const& worldSize, uint const searchRadius )
:	m_worldCells( worldCells ),
	m_worldSize( worldSize ),
	m_nSearchRadius( searchRadius )
{
	m_nCells = m_worldCells.x * m_worldCells.y * m_worldCells.z;

	// Create the cells.
	CreateCells();
}

void KNNBinDataV1::CreateCells( void )
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

				index++;
			}
		}
	}

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
	CUDA_SAFE_CALL( cudaMemcpyToSymbol( "constWorldSizeV1", &m_worldSize, sizeof(float3) ) );
	CUDA_SAFE_CALL( cudaMemcpyToSymbol( "constWorldStepV1", &step, sizeof(float3) ) );
	CUDA_SAFE_CALL( cudaMemcpyToSymbol( "constWorldStepNormalizedV1", &stepNormalized, sizeof(float3) ) );
	CUDA_SAFE_CALL( cudaMemcpyToSymbol( "constWorldCellsV1", &m_worldCells, sizeof(uint3) ) );

	// Free host memory.
	free( phCellIndices );
}
