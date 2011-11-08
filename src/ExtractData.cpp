#include "OpenSteer/ExtractData.h"

#include <vector>
#include <fstream>

#define cimg_use_tiff
#include "C:\Users\owen\Development\Libraries\CImg-1.4.9\CImg.h"

using namespace cimg_library;

using namespace OpenSteer;

extern "C"
{
	void ComputeCellDensity(		AgentGroup * pAgentGroup, uint * pAgentsPerCell );
	void ComputeAvgCellVelocity(	AgentGroup * pAgentGroup, float3 * pAvgCellVelocity );
}

void OpenSteer::WriteCellDensity( char const* szAllCellsFilenamePrefix, AgentGroup * pAgentGroup, std::vector< uint > vecSelectedCells )
{
	char szFilename[100] = {0};

	uint const&		numCells	= pAgentGroup->GetKNNDatabase().cells();
	uint3 const&	worldCells	= pAgentGroup->GetKNNDatabase().worldCells();

	// Allocate host-side memory.
	uint *			phAgentsPerCell = (uint*)malloc( numCells * sizeof(uint) );
	
	// Run the kernel.
	::ComputeCellDensity( pAgentGroup, phAgentsPerCell );

	for( uint i = 0; i < vecSelectedCells.size(); i++ )
	{
		// Open the data file for writing (selected cells)
		sprintf_s( szFilename, "%s_%d.dat", "Frames/cell_density", vecSelectedCells[i] );
		std::ofstream ofsSelectedCells;
		
		ofsSelectedCells.open( szFilename, std::ios_base::app );
		
		if( ! ofsSelectedCells.is_open() )
			ofsSelectedCells.open( szFilename );

		if( ofsSelectedCells.is_open() )
		{
			ofsSelectedCells << phAgentsPerCell[ vecSelectedCells[ i ] ] << std::endl;

			ofsSelectedCells.close();
		}
	}


	// Open the data file for writing (all cells in frame)
	sprintf_s( szFilename, "%s.dat", szAllCellsFilenamePrefix );
	std::ofstream ofsAllCells( szFilename );

	if( ofsAllCells.is_open() )
	{
		ofsAllCells << "%\tCell densities" << std::endl;
		ofsAllCells << "%\tworldCells:\tX: " << worldCells.x << "\tY: " << worldCells.y << "\tZ: " << worldCells.z << std::endl;
		ofsAllCells << "%\tData format: " << std::endl;
		ofsAllCells << "%\tX\tY\tnumAgents" << std::endl;

		for( uint i = 0; i < worldCells.z - 1; i++ )
		{
			for( uint j = 0; j < worldCells.x; j++ )
			{
				ofsAllCells << phAgentsPerCell[i*worldCells.x + j] << '\t';
			}
			ofsAllCells << std::endl;
		}

		ofsAllCells.close();
	}
/*
	// Create an image of unsigned char values.
	CImg< unsigned char > image( worldCells.x, worldCells.z );
	// Populate with the data.
	for( uint i = 0; i < worldCells.z; i++ )
		for( uint j = 0; j < worldCells.x; j++ )
			image(j,i) =  (unsigned char)min((phAgentsPerCell[i*worldCells.x + j]) * 16, 255);

	memset( szFilename, 0, 100 * sizeof(char) );
	sprintf_s( szFilename, "%s.tiff", szAllCellsFilenamePrefix );
	image.save_tiff( szFilename );
*/
	
	free( phAgentsPerCell );
}

void OpenSteer::WriteAvgCellVelocity( char const* szFilenamePrefix, AgentGroup * pAgentGroup, std::vector< uint > vecSelectedCells )
{
	char szFilename[100] = {0};

	uint3 const&	worldCells	= pAgentGroup->GetKNNDatabase().worldCells();
	uint const&		numCells	= pAgentGroup->GetKNNDatabase().cells();

	// Allocate host-side memory.
	float3 * phAvgCellVelocity = (float3*)malloc( numCells * sizeof(float3) );

	// Run the kernel.
	ComputeAvgCellVelocity( pAgentGroup, phAvgCellVelocity );

	for( uint i = 0; i < vecSelectedCells.size(); i++ )
	{
		// Open the data file for writing (selected cells)
		sprintf_s( szFilename, "%s_%d.dat", "Frames/cell_avg_velocity", vecSelectedCells[i] );
		std::ofstream ofsSelectedCells;
		
		ofsSelectedCells.open( szFilename, std::ios_base::app );
		
		if( ! ofsSelectedCells.is_open() )
			ofsSelectedCells.open( szFilename );

		if( ofsSelectedCells.is_open() )
		{
			ofsSelectedCells << phAvgCellVelocity[ vecSelectedCells[ i ] ].x << '\t' << phAvgCellVelocity[ vecSelectedCells[ i ] ].z << std::endl;
			ofsSelectedCells.close();
		}
	}

	// Open the data file for writing.
	sprintf_s( szFilename, "%s.dat", szFilenamePrefix );
	std::ofstream of( szFilename );

	if( of.is_open() )
	{
		of << "%\tCell average velocities" << std::endl;
		of << "%\tworldCells:\tX: " << worldCells.x << "\tY: " << worldCells.y << "\tZ: " << worldCells.z << std::endl;
		of << "%\tData format: " << std::endl;
		of << "%\tX\tZ\tV.x\tV.z" << std::endl;

		for( uint i = 0; i < worldCells.z - 1; i++ )
			for( uint j = 0; j < worldCells.x; j++ )
				of << i << '\t' << j << '\t' << phAvgCellVelocity[i*worldCells.z + j].x << '\t' << phAvgCellVelocity[i*worldCells.x + j].z << std::endl;

		of.close();
	}

	free( phAvgCellVelocity );
}

