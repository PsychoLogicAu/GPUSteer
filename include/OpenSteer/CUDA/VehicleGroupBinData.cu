#include "VehicleGroupBinData.cuh"

#include "CUDAGlobals.cuh"

using namespace OpenSteer;

bin_data::bin_data( uint3 const& worldCells, float3 const& worldSize )
:	m_worldCells( worldCells ),
	m_worldSize( worldSize )
{
	// Create the cells.
	CreateCells();
}

void bin_data::CreateCells( void )
{
	float3 const step = make_float3(	m_worldSize.x / m_worldCells.x,		// width
										m_worldSize.y / m_worldCells.y,		// depth
										m_worldSize.z / m_worldCells.z );	// height

/*
Texture addressing in CUDA operates as follows. The binning representation should match it internally.
   z|
	|    y/
	|    /
	|   /
	|  /
	| /
	|/_________x
*/
	// Allocate host memory to temporarily store the 3D texture data.
	uint * phCellIndices = (uint*)malloc( m_worldCells.x * m_worldCells.y * m_worldCells.z * sizeof(uint) );

	uint offset = 0;

	for( size_t z = 0; z < m_worldCells.z; z++ )			// height - z axis
	{
		for( size_t y = 0; y < m_worldCells.y; y++ )		// depth - y axis
		{
			for( size_t x = 0; x < m_worldCells.x; x++ )	// width - x axis
			{
				// Make a bin_cell structure.
				bin_cell bc;

				//bc.iBinIndex = iBinIndex;
				bc.iCellIndex = x + (y * m_worldCells.x) + (z * m_worldCells.y * m_worldCells.x);

				// Set the offset value for the cell lookup texture.
				phCellIndices[offset++] = bc.iCellIndex;

				// TODO: set uint3 indices of m_neighborPosMin & m_neighborPosMax (?)

				// Cell is initially empty.
				bc.iBegin = 0;
				bc.iEnd = 0;
				bc.nSize = 0;

				// Set the minBounds of the cell.
				bc.minBounds.x = x * step.x;
				bc.minBounds.y = y * step.y;
				bc.minBounds.z = z * step.z;

				// Set the maxBounds of the cell.
				bc.maxBounds.x = bc.minBounds.x + step.x;
				bc.maxBounds.y = bc.minBounds.y + step.y;
				bc.maxBounds.z = bc.minBounds.z + step.z;

				m_hvCells.push_back( bc );
			}
		}
	}

	// Transfer the bin_cell structures to the device memory.
	m_dvCells = m_hvCells;

	// Prepare bin_cell index lookup texture.

	// Set the extents for the 3D texture memory.
	cudaExtent const worldSize = make_cudaExtent( m_worldCells.x, m_worldCells.y, m_worldCells.z );

	// Create the channel desc.
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc< uint >();

	// Allocate m_pdCellIndexArray.
	CUDA_SAFE_CALL( cudaMalloc3DArray( &m_pdCellIndexArray, &channelDesc, worldSize ) );

	// Copy data to 3D array.
	cudaMemcpy3DParms copyParms = {0};
	copyParms.srcPtr = make_cudaPitchedPtr( (void*)phCellIndices, worldSize.width*sizeof(uint), worldSize.width, worldSize.height );
	copyParms.dstArray = m_pdCellIndexArray;
	copyParms.extent = worldSize;
	copyParms.kind = cudaMemcpyHostToDevice;
	CUDA_SAFE_CALL( cudaMemcpy3D( &copyParms ) );

	// Free dynamic memory.
	free( phCellIndices );
}