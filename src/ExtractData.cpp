#include "OpenSteer/ExtractData.h"

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

void OpenSteer::WriteCellDensity( char const* szFilenamePrefix, AgentGroup * pAgentGroup )
{
	char szFilename[100] = {0};

	uint const&		numCells	= pAgentGroup->GetKNNDatabase().cells();
	uint3 const&	worldCells	= pAgentGroup->GetKNNDatabase().worldCells();

	// Allocate host-side memory.
	uint *			phAgentsPerCell = (uint*)malloc( numCells * sizeof(uint) );
	
	// Run the kernel.
	::ComputeCellDensity( pAgentGroup, phAgentsPerCell );

	// Open the data file for writing.
	sprintf_s( szFilename, "%s.dat", szFilenamePrefix );
	std::ofstream of( szFilename );

	if( of.is_open() )
	{
		of << "%\tCell densities" << std::endl;
		of << "%\tworldCells:\tX: " << worldCells.x << "\tY: " << worldCells.y << "\tZ: " << worldCells.z << std::endl;
		of << "%\tData format: " << std::endl;
		of << "%\tX\tY\tnumAgents" << std::endl;

		for( uint i = 0; i < worldCells.x; i++ )
			for( uint j = 0; j < worldCells.z; j++ )
				of << i << '\t' << j << '\t' << phAgentsPerCell[i*worldCells.x + j] << std::endl;

		of.close();
	}

	// Create an image of unsigned char values.
	CImg< unsigned char > image( worldCells.x, worldCells.z );
	// Populate with the data.
	for( uint i = 0; i < worldCells.x; i++ )
		for( uint j = 0; j < worldCells.z; j++ )
			image(i,j) =  (unsigned char)(phAgentsPerCell[i*worldCells.x + j]) * 60;

	memset( szFilename, 0, 100 * sizeof(char) );
	sprintf_s( szFilename, "%s.tiff", szFilenamePrefix );
	image.save_tiff( szFilename );
	
	free( phAgentsPerCell );
}

void OpenSteer::WriteAvgCellVelocity( char const* szFilenamePrefix, AgentGroup * pAgentGroup )
{
	char szFilename[100] = {0};

	uint3 const&	worldCells	= pAgentGroup->GetKNNDatabase().worldCells();
	uint const&		numCells	= pAgentGroup->GetKNNDatabase().cells();

	// Allocate host-side memory.
	float3 * phAvgCellVelocity = (float3*)malloc( numCells * sizeof(float3) );

	// Run the kernel.
	ComputeAvgCellVelocity( pAgentGroup, phAvgCellVelocity );

	// Open the data file for writing.
	sprintf_s( szFilename, "%s.dat", szFilenamePrefix );
	std::ofstream of( szFilename );

	if( of.is_open() )
	{
		of << "%\tCell average velocities" << std::endl;
		of << "%\tworldCells:\tX: " << worldCells.x << "\tY: " << worldCells.y << "\tZ: " << worldCells.z << std::endl;
		of << "%\tData format: " << std::endl;
		of << "%\tX\tY\tV.x\tV.y" << std::endl;

		for( uint i = 0; i < worldCells.x; i++ )
			for( uint j = 0; j < worldCells.z; j++ )
				of << i << '\t' << j << '\t' << phAvgCellVelocity[i*worldCells.x + j].x << '\t' << phAvgCellVelocity[i*worldCells.x + j].z << std::endl;

		of.close();
	}

	free( phAvgCellVelocity );
}

