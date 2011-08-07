#include "VehicleGroupBinData.cuh"

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
	float3 const step = make_float3( 
	float const fStepX = m_fWorldSizeX / m_nCellsX;
	float const fStepY = m_fWorldSizeY / m_nCellsY;
	float const 

	for( size_t iRow = 0; iRow < m_nCellsX; iRow++ )
	{
		for( size_t iCol = 0; iCol < m_nCellsY; iCol++ )
		{
			bin_cell bc;

			// Set the data for the bin_cell.
			bc.iBinIndex = iRow * m_nCellsX + iCol;
			bc.iBegin = 0;
			bc.iEnd = 0;
			bc.nSize = 0;
			// Set the min and max coordinates of the bin_cell.
			bc.minXY.x = iRow * fStepX;
			bc.minXY.y = iCol * fStepY;

			bc.maxXY.x = iRow * fStepX + fStepX;
			bc.maxXY.y = iCol * fStepY + fStepY;

			m_hvCells.push_back( bc );
		}
	}

	// Transfer the data to the device memory.
	m_dvCells = m_hvCells;
}