#include "OpenSteer/VehicleGroup.h"

#include "OpenSteer/VectorUtils.cuh"
#include "OpenSteer/Utilities.h"

using namespace OpenSteer;

VehicleGroup::VehicleGroup( uint3 const& worldCells, uint const knn )
:	BaseGroup( knn, 0, worldCells.x*worldCells.y*worldCells.z ),
	m_nCount( 0 )
{
}

VehicleGroup::~VehicleGroup(void)
{
	Clear();
}

// Gets the index of a vehicle in the vector from its id number.
int VehicleGroup::GetVehicleIndex( unsigned int _id ) const
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
bool VehicleGroup::AddVehicle( VehicleData const& vd, VehicleConst const& vc )
{
	// Add the vehicle if it is not already contained.
	if( GetVehicleIndex( vc.id ) == -1 )
	{
		// Add the vehicle's data to the host structures.
		m_vehicleGroupData.addVehicle( vd );
		m_vehicleGroupConst.addVehicle( vc );

		// Will need to sync the device.
		m_vehicleGroupData.m_bSyncDevice = true;
		m_vehicleGroupConst.m_bSyncDevice = true;

		// Add the id and index to the IDToIndexMap.
		m_cIDToIndexMap[ vc.id ] = m_nCount++;

		// Resize the neighbor database.
		m_neighborDB.resize( m_nCount );

		return true;
	}

	return false;
}

// Removes a vehicle from the group using the supplied id number.
void VehicleGroup::RemoveVehicle( id_type const id )
{
	// Get the vehicle's index.
	int index = GetVehicleIndex( id );

	if(index > -1) // Found.
	{
		// Remove the vehicle from the host structures.
		m_vehicleGroupConst.removeVehicle( index );
		m_vehicleGroupData.removeVehicle( index );

		// There is now one less.
		m_nCount--;

		// Resize the neighbor database.
		m_neighborDB.resize( m_nCount );

		// The vehicle indices will have changed. Rebuild the map.
		RebuildIDToIndexMap();

		// Will need to sync the device.
		m_vehicleGroupData.m_bSyncDevice = true;
		m_vehicleGroupConst.m_bSyncDevice = true;
	}
}

void VehicleGroup::RebuildIDToIndexMap( void )
{
	// Clear the current map.
	m_cIDToIndexMap.clear();

	size_t index = 0;
	// For each vehicle ID in the host hvId vector...
	for(	std::vector<id_type>::const_iterator it = m_vehicleGroupConst.hvId().begin();
			it != m_vehicleGroupConst.hvId().end();
			++it, ++index )
	{
		m_cIDToIndexMap[ (*it) ] = index;
	}
}

void VehicleGroup::Clear(void)
{
	// Clear the host and device vectors.
	m_vehicleGroupData.clear();
	m_vehicleGroupConst.clear();

	// Clear the neighbor database.
	m_neighborDB.clear();

	m_cIDToIndexMap.clear();

	m_nCount = 0;
}

// Gets the data for a vehicle from the supplied id.
bool VehicleGroup::GetDataForVehicle( id_type const id, VehicleData & vd, VehicleConst & vc )
{
	// Do I want to do synchronization from this class, or vehicle_group_data/vehicle_group_const ???
	// Sync the host.
	SyncHost();

	// Get the index of the vehicle.
	int index = GetVehicleIndex( id );

	if( index == -1 ) // Not found.
		return false;

	m_vehicleGroupConst.getVehicleData( index, vc );
	m_vehicleGroupData.getVehicleData( index, vd );

	return true;
}

void VehicleGroup::SyncDevice( void )
{
	m_vehicleGroupData.syncDevice();
	m_vehicleGroupConst.syncDevice();

	m_neighborDB.syncDevice();
}

void VehicleGroup::SyncHost( void )
{
	m_vehicleGroupData.syncHost();
	m_vehicleGroupConst.syncHost();

	m_neighborDB.syncHost();
}
