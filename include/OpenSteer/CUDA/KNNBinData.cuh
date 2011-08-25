#ifndef OPENSTEER_VEHICLEGROUPBINDATA_CUH
#define OPENSTEER_VEHICLEGROUPBINDATA_CUH

#include "dev_vector.cuh"

#include "CUDAGlobals.cuh"

#include <vector>

namespace OpenSteer
{

struct bin_cell
{
	size_t	iCellIndex;	// Index of this cell.
	size_t	iBegin;		// Index of first vehicle in this cell.
	size_t	iEnd;		// Index of last vehicle in this cell.
	size_t	nSize;		// Number of vehicles in this cell.
	float3	minBounds;	// Minimum bounds of this cell.
	float3	maxBounds;	// Maximum bounds of this cell.
	// TODO: uint3 neighborPosMin & neighborPosMax (?)

	// Returns true if pPosition is within the bounds of this cell. false otherwise.
	__host__ __device__ bool Within( float3 const* pPosition )
	{
		if( pPosition->x > minBounds.x && pPosition->x <= maxBounds.x &&
			pPosition->y > minBounds.y && pPosition->y <= maxBounds.y &&
			pPosition->z > minBounds.z && pPosition->z <= maxBounds.z )
			return true;

		return false;
	}
};

class bin_data
{
private:
	// worldCells, number of cells in each dimension. z is up!
	uint3		m_worldCells;
	// worldSize, extent of the world in each dimension. z is up!
	float3		m_worldSize;
	// total number of cells.
	uint		m_nCells;

	dev_vector<bin_cell>	m_dvCells;
	// Host vector used while creating the bin_cell structures..
	std::vector<bin_cell>	m_hvCells;

	// cudaArray used to hold the bin_cell structures on the device.
	cudaArray *				m_pdCellIndexArray;

	void CreateCells( void );

public:
	//bin_data(	size_t const nCellsX, size_t const nCellsY, float const fWorldSizeX, float const fWorldSizeY );
	bin_data( uint3 const& worldCells, float3 const& worldSize );
	~bin_data( void ) {}

	bin_cell *	pdBinCells( void )			{ return m_dvCells.begin(); }
	cudaArray *	pdCellIndexArray( void )	{ return m_pdCellIndexArray; }
	//size_t		Size( void )		{ return m_nSize; }

	// Get methods for the number of cells and the world size.
	uint3 const& WorldCells( void ) const	{ return m_worldCells; }
	float3 const& WorldSize( void ) const	{ return m_worldSize; }
	uint getNumCells( void ) const			{ return m_nCells; }
};	// class bin_data
typedef bin_data BinData;

}	// namespace OpenSteer
#endif

