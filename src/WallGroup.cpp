#include "OpenSteer/WallGroup.h"

using namespace OpenSteer;

WallGroup::WallGroup( uint3 const& worldCells, uint const knw )
:	BaseGroup( knw, 0, worldCells )
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

void WallGroup::SplitWalls( std::vector< bin_cell > const& cells )
{
	m_wallGroupData.SplitWalls( cells );

	m_neighborDB.resize( Size() );
}
