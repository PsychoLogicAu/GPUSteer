#ifndef OPENSTEER_VEHICLEGROUP_H
#define OPENSTEER_VEHICLEGROUP_H

#include "VehicleGroupData.h"
#include <map>

namespace OpenSteer
{
class VehicleGroup
{
	typedef std::map< id_type, size_t > IDToIndexMap;

protected:
	int		GetVehicleIndex( id_type _id ) const;

	void	RebuildIDToIndexMap( void );
	/// Copy the host structures to the device.
	void	SyncDevice( void );
	/// Copy the device structures to the host.
	void	SyncHost( void );


	// Host data.
	VehicleGroupDataHost		m_vehicleDataHost;
	VehicleGroupConstHost		m_vehicleConstHost;

	// Device data.
	VehicleGroupDataDevice		m_vehicleDataDevice;
	VehicleGroupConstDevice		m_vehicleConstDevice;

	IDToIndexMap				m_cIDToIndexMap;

	size_t						m_nCount;

	bool						m_bSyncDevice;
	bool						m_bSyncHost;

public:
	VehicleGroup( void );
	virtual ~VehicleGroup( void );

	bool AddVehicle( VehicleData const& vd, VehicleConst const& vc );
	void RemoveVehicle( id_type const id );
	/// Clear all vehicles from the group.
	void Clear( void );
	void SetSyncHost( void )	{ m_bSyncHost = true; }

	/// Get the size of the collection.
	size_t Size( void ) const { return m_nCount; }

	VehicleGroupConstDevice &	GetVehicleGroupConstDevice()	{ return m_vehicleConstDevice; }
	VehicleGroupDataDevice &	GetVehicleGroupDataDevice()		{ return m_vehicleDataDevice; }
	
	VehicleGroupConstHost &		GetVehicleGroupConstHost()		{ return m_vehicleConstHost; }
	VehicleGroupDataHost &		GetVehicleGroupDataHost()		{ return m_vehicleDataHost; }

	/// Use to extract data for an individual vehicle
	bool GetDataForVehicle( id_type const id, VehicleData &_data, VehicleConst &_const);

	void OutputDataToFile( const char *filename );
};	//class VehicleGroup
}	//namespace OpenSteer
#endif
