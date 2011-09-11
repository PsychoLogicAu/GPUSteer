#ifndef OPENSTEER_VEHICLEGROUPBINDATA_CUH
#define OPENSTEER_VEHICLEGROUPBINDATA_CUH

#include "dev_vector.cuh"

#include "CUDAGlobals.cuh"
#include "CUDAKernelGlobals.cuh"

#include <vector>

namespace OpenSteer
{

struct bin_cell
{
	size_t	index;		// Index of this cell.
	float3	position;
	float3	minBound;	// Minimum bounds of this cell.
	float3	maxBound;	// Maximum bounds of this cell.
};

class bin_data
{
private:
	uint3					m_worldCells;			// Number of cells in each world dimension.
	float3					m_worldSize;			// Size of the world in world coordinates.
	
	uint					m_nCells;				// Total number of cells.

	uint					m_nSearchRadius;		// Distance in cells to search for neighbors.
	uint					m_nNeighborsPerCell;
	dev_vector< uint >		m_dvCellNeighbors;

	std::vector< bin_cell >	m_hvCells;
	dev_vector< bin_cell >	m_dvCells;

	// cudaArray used to hold the bin_cell structures on the device.
	cudaArray *				m_pdCellIndexArray;

	void CreateCells( void );
	void ComputeCellNeighbors( bool b3D );

	virtual dim3 gridDim( void )	{	return dim3( ( getNumCells() + THREADSPERBLOCK - 1 ) / THREADSPERBLOCK );	}
	virtual dim3 blockDim( void )	{	return dim3( THREADSPERBLOCK );	}

public:
	bin_data( uint3 const& worldCells, float3 const& worldSize, uint const searchRadius );
	~bin_data( void ) {}

	uint		radius( void )				{ return m_nSearchRadius; }
	uint		neighborsPerCell( void )	{ return m_nNeighborsPerCell; }

	// Get methods for device data.
	uint *		pdCellNeighbors( void )		{ return m_dvCellNeighbors.begin(); }
	bin_cell *	pdCells( void )				{ return m_dvCells.begin(); }
	cudaArray *	pdCellIndexArray( void )	{ return m_pdCellIndexArray; }

	// Get methods for the number of cells and the world size.
	uint3 const& WorldCells( void ) const	{ return m_worldCells; }
	float3 const& WorldSize( void ) const	{ return m_worldSize; }
	uint getNumCells( void ) const			{ return m_nCells; }
};	// class bin_data
typedef bin_data BinData;

}	// namespace OpenSteer
#endif

