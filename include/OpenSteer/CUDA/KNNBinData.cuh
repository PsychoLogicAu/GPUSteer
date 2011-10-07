#ifndef OPENSTEER_VEHICLEGROUPBINDATA_CUH
#define OPENSTEER_VEHICLEGROUPBINDATA_CUH

#include "dev_vector.cuh"

#include "CUDAGlobals.cuh"
#include "CUDAKernelGlobals.cuh"

#include <vector>

#define KNN_THREADSPERBLOCK 128

namespace OpenSteer
{

//struct bin_cell
//{
//	size_t	index;		// Index of this cell.
//	float3	position;
//	float3	minBound;	// Minimum bounds of this cell.
//	float3	maxBound;	// Maximum bounds of this cell.
//};

class KNNBinData
{
private:
	uint3					m_worldCells;			// Number of cells in each world dimension.
	float3					m_worldSize;			// Size of the world in world coordinates.

	float3					m_worldStep;
	float3					m_worldStepNormalized;
	
	uint					m_nCells;				// Total number of cells.

	uint					m_nSearchRadius;		// Distance in cells to search for neighbors.
	uint					m_nNeighborsPerCell;

	dev_vector< float3 >	m_dvPositions;
	dev_vector< float3 >	m_dvMinBounds;
	dev_vector< float3 >	m_dvMaxBounds;

	std::vector< float3 >	m_hvPositions;
	std::vector< float3 >	m_hvMinBounds;
	std::vector< float3 >	m_hvMaxBounds;

	// cudaArray used to hold the bin_cell structures on the device.
	cudaArray *				m_pdCellIndexArray;

	void CreateCells( void );
	void ComputeCellNeighbors( void );
	void ComputeCellNeighbors2D( float3 * pNeighborOffsets );
	void ComputeCellNeighbors3D( float3 * pNeighborOffsets );

	virtual dim3 gridDim( void )	{	return dim3( ( getNumCells() + KNN_THREADSPERBLOCK - 1 ) / KNN_THREADSPERBLOCK );	}
	virtual dim3 blockDim( void )	{	return dim3( KNN_THREADSPERBLOCK );	}

public:
	KNNBinData( uint3 const& worldCells, float3 const& worldSize, uint const searchRadius );
	~KNNBinData( void ) {}

	uint		radius( void )									{ return m_nSearchRadius; }
	uint		neighborsPerCell( void )						{ return m_nNeighborsPerCell; }

	// Get methods for device data.
	float3 *	pdCellPositions( void )							{ return m_dvPositions.begin(); }
	float3 *	pdCellMinBounds( void )							{ return m_dvMinBounds.begin(); }
	float3 *	pdCellMaxBounds( void )							{ return m_dvMaxBounds.begin(); }
	cudaArray *	pdCellIndexArray( void )						{ return m_pdCellIndexArray; }

	// Get methods for host data.
	std::vector< float3 > const& hvCellPositions( void ) const	{ return m_hvPositions; }
	std::vector< float3 > const& hvCellMinBounds( void ) const	{ return m_hvMinBounds; }
	std::vector< float3 > const& hvCellMaxBounds( void ) const	{ return m_hvMaxBounds; }


	// Get methods for the number of cells and the world size.
	uint3 const& WorldCells( void ) const						{ return m_worldCells; }
	float3 const& WorldSize( void ) const						{ return m_worldSize; }
	uint getNumCells( void ) const								{ return m_nCells; }
};	// class bin_data
}	// namespace OpenSteer
#endif

