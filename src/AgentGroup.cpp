#include "OpenSteer/AgentGroup.h"

#include "OpenSteer/VectorUtils.cuh"
#include "OpenSteer/Utilities.h"

using namespace OpenSteer;

AgentGroup::AgentGroup( uint3 const& worldCells, uint const knn )
:	BaseGroup( knn, 0, worldCells )
{
}

AgentGroup::~AgentGroup(void)
{
	Clear();
}

// Gets the index of a vehicle in the vector from its id number.
int AgentGroup::GetAgentIndex( uint const id ) const
{
	int index = -1;

	// Check if m_cIDToIndexMap contains the key.
	IDToIndexMap::const_iterator it = m_cIDToIndexMap.find( id );
	if( it != m_cIDToIndexMap.end() )
	{
		index = it->second;
	}

	return index;
}

// Adds a vehicle to the group.
bool AgentGroup::AddAgent( AgentData const& ad )
{
	// Add the vehicle if it is not already contained.
	if( GetAgentIndex( ad.id ) == -1 )
	{
		// Add the agent's data to the host structures.
		m_agentGroupData.addAgent( ad );

		// Will need to sync the device.
		m_agentGroupData.m_bSyncDevice = true;

		// Add the id and index to the IDToIndexMap.
		m_cIDToIndexMap[ ad.id ] = m_nCount++;

		// Resize the neighbor database.
		m_neighborDB.resize( m_nCount );

		return true;
	}

	return false;
}

// Removes a vehicle from the group using the supplied id number.
void AgentGroup::RemoveAgent( id_type const id )
{
	// Get the vehicle's index.
	int index = GetAgentIndex( id );

	if(index > -1) // Found.
	{
		// Remove the vehicle from the host structures.
		m_agentGroupData.removeAgent( index );

		// There is now one less.
		m_nCount--;

		// Resize the neighbor database.
		m_neighborDB.resize( m_nCount );

		// The vehicle indices will have changed. Rebuild the map.
		RebuildIDToIndexMap();

		// Will need to sync the device.
		m_agentGroupData.m_bSyncDevice = true;
	}
}

void AgentGroup::RebuildIDToIndexMap( void )
{
	// Clear the current map.
	m_cIDToIndexMap.clear();

	size_t index = 0;
	// For each vehicle ID in the host hvId vector...
	for(	std::vector<id_type>::const_iterator it = m_agentGroupData.hvID().begin();
			it != m_agentGroupData.hvID().end();
			++it, ++index )
	{
		m_cIDToIndexMap[ (*it) ] = index;
	}
}

void AgentGroup::Clear(void)
{
	// Clear the host and device vectors.
	m_agentGroupData.clear();

	// Clear the neighbor database.
	m_neighborDB.clear();

	m_cIDToIndexMap.clear();

	m_nCount = 0;
}

// Gets the data for a vehicle from the supplied id.
bool AgentGroup::GetDataForAgent( id_type const id, AgentData & ad )
{
	// Do I want to do synchronization from this class, or vehicle_group_data/vehicle_group_const ???
	// Sync the host.
	SyncHost();

	// Get the index of the vehicle.
	int index = GetAgentIndex( id );

	if( index == -1 ) // Not found.
		return false;

	m_agentGroupData.getAgentData( index, ad );

	return true;
}

void AgentGroup::SyncDevice( void )
{
	m_agentGroupData.syncDevice();

	m_neighborDB.syncDevice();
}

void AgentGroup::SyncHost( void )
{
	m_agentGroupData.syncHost();

	m_neighborDB.syncHost();
}
