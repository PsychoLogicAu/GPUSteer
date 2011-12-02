#ifndef OPENSTEER_KNNBINDATAV2_CUH
#define OPENSTEER_KNNBINDATAV2_CUH

#include "KNNBinDataGlobals.cuh"

#include "dev_vector.cuh"

#include "CUDAGlobals.cuh"
#include "CUDAKernelGlobals.cuh"

#include <vector>

#define KNN_THREADSPERBLOCK 128

namespace OpenSteer
{

class KNNBinDataV3
{
private:
	uint3					m_worldCells;			// Number of cells in each world dimension.
	float3					m_worldSize;			// Size of the world in world coordinates.
	
	uint					m_nCells;				// Total number of cells.

	uint					m_nSearchRadius;		// Distance in cells to search for neighbors.
	uint					m_nNeighborsPerCell;
	dev_vector< uint >		m_dvCellNeighbors;

	void ComputeCellNeighbors( bool b3D );
	uint calcGridHash( int3 const& gridPos );

	virtual dim3 gridDim( void )	{	return dim3( ( getNumCells() + KNN_THREADSPERBLOCK - 1 ) / KNN_THREADSPERBLOCK );	}
	virtual dim3 blockDim( void )	{	return dim3( KNN_THREADSPERBLOCK );	}

public:
	KNNBinDataV3( uint3 const& worldCells, float3 const& worldSize, uint const searchRadius );
	~KNNBinDataV3( void ) {}

	uint		radius( void )								{ return m_nSearchRadius; }
	uint		neighborsPerCell( void )					{ return m_nNeighborsPerCell; }

	// Get methods for device data.
	uint *		pdCellNeighbors( void )						{ return m_dvCellNeighbors.begin(); }

	// Get methods for the number of cells and the world size.
	uint3 const& WorldCells( void ) const					{ return m_worldCells; }
	float3 const& WorldSize( void ) const					{ return m_worldSize; }
	uint getNumCells( void ) const							{ return m_nCells; }
	bool is3D( void ) const									{ return m_worldCells.y > 1; }
};	// class KNNBinDataV3
}	// namespace OpenSteer
#endif
