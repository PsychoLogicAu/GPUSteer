#include "OpenSteer/ObstacleGroup.h"

#include "OpenSteer/VectorUtils.cuh"
#include "OpenSteer/Utilities.h"

using namespace OpenSteer;

ObstacleGroup::ObstacleGroup( uint3 const& worldCells, uint const kno )
:	BaseGroup( kno, 0, worldCells ),
	m_nCount( 0 )
{
}

ObstacleGroup::~ObstacleGroup(void)
{
	Clear();
}

// Adds a vehicle to the group.
void ObstacleGroup::AddObstacle(  ObstacleData const& od )
{
	// Add the vehicle's data to the host structures.
	m_obstacleGroupData.addObstacle( od );

	// There is now one more.
	m_nCount++;

	// Resize the neighbor database.
	m_neighborDB.resize( m_nCount );
}

// Removes a vehicle from the group using the supplied id number.
void ObstacleGroup::RemoveObstacle( uint const index )
{
	if( index < m_nCount )
	{
		// Remove the vehicle from the host structures.
		m_obstacleGroupData.removeObstacle( index );

		// There is now one less.
		m_nCount--;

		// Resize the neighbor database.
		m_neighborDB.resize( m_nCount );

		// Will need to sync the device.
		m_obstacleGroupData.m_bSyncDevice = true;
	}
}

void ObstacleGroup::Clear( void )
{
	// Clear the host and device vectors.
	m_obstacleGroupData.clear();

	// Clear the neighbor database.
	m_neighborDB.clear();

	m_nCount = 0;
}

// Gets the data for a vehicle from the supplied id.
bool ObstacleGroup::GetDataForObstacle( uint const index, ObstacleData & od )
{
	if( index < m_nCount )
	{
		m_obstacleGroupData.getObstacleData( index, od );

		return true;
	}

	return false;
}

void ObstacleGroup::SyncDevice( void )
{
	m_obstacleGroupData.syncDevice();

	m_neighborDB.syncDevice();
}
