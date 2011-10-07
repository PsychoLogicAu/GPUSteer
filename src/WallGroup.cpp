#include "OpenSteer/WallGroup.h"

using namespace OpenSteer;

WallGroup::WallGroup( uint3 const& worldCells, uint const knw )
:	BaseGroup( knw, 0, worldCells.x*worldCells.y*worldCells.z )
{
	// Nothing to do.
}

void WallGroup::SyncDevice( void )
{
	m_neighborDB.syncDevice();
}

bool WallGroup::LoadFromFile( char const* szFilename )
{
	return m_wallGroupData.LoadFromFile( szFilename );
}

void WallGroup::SplitWalls( std::vector< float3 > const& cellMinBounds, std::vector< float3 > const& cellMaxBounds )
{
	m_wallGroupData.SplitWalls( cellMinBounds, cellMaxBounds );

	m_neighborDB.resize( Size() );
}
