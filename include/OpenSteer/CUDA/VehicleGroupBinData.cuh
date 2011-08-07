#ifndef OPENSTEER_VEHICLEGROUPBINDATA_CUH
#define OPENSTEER_VEHICLEGROUPBINDATA_CUH

#include "dev_vector.cuh"

#include <vector>

namespace OpenSteer
{

struct bin_cell
{
	size_t	iBinIndex;	// Index of this cell.
	size_t	iBegin;		// Index of first vehicle in this cell.
	size_t	iEnd;		// Index of last vehicle in this cell.
	size_t	nSize;		// Number of vehicles in this cell.
	float3	minBounds;	// Minimum bounds of this cell.
	float3	maxBounds;	// Maximum bounds of this cell.
};

class bin_data
{
private:
	//bool		m_bSyncHost;	// Host data needs to be synchronized.
	//bool		m_bSyncDevice;	// Device data needs to be synchronized.
	//size_t		m_nSize;		// Number of vehicles in the database.

	//size_t		m_nCellsX;		// Number of cells in the X dimension.
	//size_t		m_nCellsY;		// Number of cells in the Y dimension.
	//float		m_fWorldSizeX;	// Size of the simulation world in the X dimension.
	//float		m_fWorldSizeY;	// Size of the simulation world in the Y dimension.

	uint3		m_worldCells;
	float3		m_worldSize;

	dev_vector<bin_cell>	m_dvCells;
	std::vector<bin_cell>	m_hvCells;

	void CreateCells( void );

public:
	//bin_data(	size_t const nCellsX, size_t const nCellsY, float const fWorldSizeX, float const fWorldSizeY );
	bin_data( uint3 const& worldCells, float3 const& worldSize );
	~bin_data( void ) {}

	bin_cell *	pdBinCells( void )	{ return m_dvCells.begin(); }
	//size_t		Size( void )		{ return m_nSize; }

	// Get methods for the number of cells and the world size.
	size_t CellsX( void ) const		{ return m_nCellsX; }
	size_t CellsY( void ) const		{ return m_nCellsY; }
	float WorldSizeX( void ) const	{ return m_fWorldSizeX; }
	float WorldSizeY( void ) const	{ return m_fWorldSizeY; }
};	// class bin_data
typedef bin_data BinData;

}	// namespace OpenSteer
#endif

