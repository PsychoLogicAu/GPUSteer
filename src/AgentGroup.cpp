#include "OpenSteer/AgentGroup.h"

#include "OpenSteer/VectorUtils.cuh"
#include "OpenSteer/Utilities.h"

using namespace OpenSteer;

AgentGroup::AgentGroup( uint3 const& worldCells, uint const knn )
:	BaseGroup( knn, 0, worldCells.x*worldCells.y*worldCells.z )
{
}

AgentGroup::~AgentGroup(void)
{
	Clear();
}

// Gets the index of a vehicle in the vector from its id number.
int AgentGroup::GetVehicleIndex( unsigned int _id ) const
{
	int index = -1;

	// Check if m_cIDToIndexMap contains the key.
	IDToIndexMap::const_iterator it = m_cIDToIndexMap.find( _id );
	if( it != m_cIDToIndexMap.end() )
	{
		index = it->second;
	}

	return index;
}

// Adds a vehicle to the group.
bool AgentGroup::AddVehicle( VehicleData const& vd, VehicleConst const& vc )
{
	// Add the vehicle if it is not already contained.
	if( GetVehicleIndex( vc.id ) == -1 )
	{
		// Add the vehicle's data to the host structures.
		m_agentGroupData.addVehicle( vd );
		m_agentGroupConst.addVehicle( vc );

		// Will need to sync the device.
		m_agentGroupData.m_bSyncDevice = true;
		m_agentGroupConst.m_bSyncDevice = true;

		// Add the id and index to the IDToIndexMap.
		m_cIDToIndexMap[ vc.id ] = m_nCount++;

		// Resize the neighbor database.
		m_neighborDB.resize( m_nCount );

		return true;
	}

	return false;
}

// Removes a vehicle from the group using the supplied id number.
void AgentGroup::RemoveVehicle( id_type const id )
{
	// Get the vehicle's index.
	int index = GetVehicleIndex( id );

	if(index > -1) // Found.
	{
		// Remove the vehicle from the host structures.
		m_agentGroupConst.removeVehicle( index );
		m_agentGroupData.removeVehicle( index );

		// There is now one less.
		m_nCount--;

		// Resize the neighbor database.
		m_neighborDB.resize( m_nCount );

		// The vehicle indices will have changed. Rebuild the map.
		RebuildIDToIndexMap();

		// Will need to sync the device.
		m_agentGroupData.m_bSyncDevice = true;
		m_agentGroupConst.m_bSyncDevice = true;
	}
}

void AgentGroup::RebuildIDToIndexMap( void )
{
	// Clear the current map.
	m_cIDToIndexMap.clear();

	size_t index = 0;
	// For each vehicle ID in the host hvId vector...
	for(	std::vector<id_type>::const_iterator it = m_agentGroupConst.hvId().begin();
			it != m_agentGroupConst.hvId().end();
			++it, ++index )
	{
		m_cIDToIndexMap[ (*it) ] = index;
	}
}

void AgentGroup::Clear(void)
{
	// Clear the host and device vectors.
	m_agentGroupData.clear();
	m_agentGroupConst.clear();

	// Clear the neighbor database.
	m_neighborDB.clear();

	m_cIDToIndexMap.clear();

	m_nCount = 0;
}

// Gets the data for a vehicle from the supplied id.
bool AgentGroup::GetDataForVehicle( id_type const id, VehicleData & vd, VehicleConst & vc )
{
	// Do I want to do synchronization from this class, or vehicle_group_data/vehicle_group_const ???
	// Sync the host.
	SyncHost();

	// Get the index of the vehicle.
	int index = GetVehicleIndex( id );

	if( index == -1 ) // Not found.
		return false;

	m_agentGroupConst.getVehicleData( index, vc );
	m_agentGroupData.getVehicleData( index, vd );

	return true;
}

void AgentGroup::SyncDevice( void )
{
	m_agentGroupData.syncDevice();
	m_agentGroupConst.syncDevice();

	m_neighborDB.syncDevice();
}

void AgentGroup::SyncHost( void )
{
	m_agentGroupData.syncHost();
	m_agentGroupConst.syncHost();

	m_neighborDB.syncHost();
}
