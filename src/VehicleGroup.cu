#include "OpenSteer/VehicleGroup.h"

#ifndef _WINDOWS_
	#define WIN32_LEAN_AND_MEAN
	#include <windows.h>
#endif

// For writing the data to an output file.
#include <iostream>
#include <fstream>

#include "OpenSteer/VectorUtils.cu"
#include "OpenSteer/Utilities.h"

using namespace OpenSteer;

VehicleGroup::VehicleGroup(void)
:	m_nCount( 0 ),
	m_bSyncDevice( false ),
	m_bSyncHost( false )
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
		m_vehicleDataHost.AddVehicle( vd );
		m_vehicleConstHost.AddVehicle( vc );

		// We will need to sync the device before next simulation step.
		m_bSyncDevice = true;

		// Add the id and index to the IDToIndexMap.
		m_cIDToIndexMap[ vc.id ] = m_nCount++;

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
		m_vehicleConstHost.RemoveVehicle( index );
		m_vehicleDataHost.RemoveVehicle( index );

		// There is now one less.
		m_nCount--;

		// The vehicle indices will have changed. Rebuild the map.
		RebuildIDToIndexMap();

		// Will need to sync the device.
		m_bSyncDevice = true;
	}
}

void VehicleGroup::RebuildIDToIndexMap( void )
{
	// Clear the current map.
	m_cIDToIndexMap.clear();

	size_t index = 0;
	// For each vehicle ID in the host hvId vector...
	for(	thrust::host_vector<id_type>::iterator it = m_vehicleConstHost.hvId.begin();
			it != m_vehicleConstHost.hvId.end();
			++it, ++index )
	{
		m_cIDToIndexMap[ (*it) ] = index;
	}
}

void VehicleGroup::Clear(void)
{
	// Clear the host and device vectors.
	m_vehicleDataHost.clear();
	m_vehicleConstHost.clear();

	m_vehicleDataDevice.clear();
	m_vehicleConstDevice.clear();

	m_cIDToIndexMap.clear();

	m_nCount = 0;
}

// Gets the data for a vehicle from the supplied id.
bool VehicleGroup::GetDataForVehicle( id_type const id, VehicleData & vd, VehicleConst & vc )
{
	// Sync the host.
	SyncHost();

	// Get the index of the vehicle.
	int index = GetVehicleIndex( id );

	if( index == -1 ) // Not found.
		return false;

	m_vehicleConstHost.GetVehicleData( index, vc );
	m_vehicleDataHost.GetVehicleData( index, vd );

	return true;
}

void VehicleGroup::SyncDevice( void )
{
	if( m_bSyncDevice )
	{
		m_vehicleConstDevice = m_vehicleConstHost;
		m_vehicleDataDevice = m_vehicleDataHost;
		m_bSyncDevice = false;
	}
}

void VehicleGroup::SyncHost( void )
{
	if( m_bSyncHost )
	{
		m_vehicleConstHost = m_vehicleConstDevice;
		m_vehicleDataHost = m_vehicleDataDevice;
		m_bSyncHost = false;
	}
}


void VehicleGroup::OutputDataToFile( char const* szFilename )
{
	SyncHost();

	using std::endl;
	std::ofstream out;
	out.open( szFilename );
	if( out.is_open() )
	{
		VehicleData vd;
		VehicleConst vc;

		// For each vehicle in the group...
		for( IDToIndexMap::iterator it = m_cIDToIndexMap.begin(); it != m_cIDToIndexMap.end(); ++it )
		{
			// Get the vehicle_data and vehicle_const structures.
			size_t const& index = it->second;
			m_vehicleConstHost.GetVehicleData( index, vc );
			m_vehicleDataHost.GetVehicleData( index, vd );

			// Output the data to the stream.
			out << "id: " << vc.id << endl;
			out << "Data:" << endl;
			out << "forward: " << vd.forward << endl;
			out << "position: " << vd.position << endl;
			out << "side: " << vd.side << endl;
			out << "speed: " << vd.speed << endl;
			out << "steering: " << vd.steering << endl;
			out << "up: " << vd.up << endl;

			out << "Const:" << endl;
			out << "id: " << vc.id << endl;
			out << "mass: " << vc.mass << endl;
			out << "maxForce: " << vc.maxForce << endl;
			out << "maxSpeed: " << vc.maxSpeed << endl;
			out << "radius: " << vc.radius << endl;
			out << endl;
		}

		//for(unsigned int i = 0; i < Size(); i++)
		//{
		//	GetDataForVehicle( 
		//	const VehicleData &vdata = m_vehicleData[i];
		//	const VehicleConst &vconst = m_vehicleConst[i];

		//	out << "id: " << vconst.id << endl;
		//	out << "Data:" << endl;
		//	out << "forward: " << vdata.forward << endl;
		//	out << "position: " << vdata.position << endl;
		//	out << "side: " << vdata.side << endl;
		//	out << "speed: " << vdata.speed << endl;
		//	out << "steering: " << vdata.steering << endl;
		//	out << "up: " << vdata.up << endl;

		//	out << "Const:" << endl;
		//	out << "id: " << vconst.id << endl;
		//	out << "mass: " << vconst.mass << endl;
		//	out << "maxForce: " << vconst.maxForce << endl;
		//	out << "maxSpeed: " << vconst.maxSpeed << endl;
		//	out << "radius: " << vconst.radius << endl;
		//	out << endl;
		//}

		out.close();
	}
}
