#ifndef OPENSTEER_VEHICLEGROUP_H
#define OPENSTEER_VEHICLEGROUP_H

#include "BaseGroup.h"

#include "VehicleGroupData.cuh"
#include "CUDA/KNNBinData.cuh"
#include "CUDA/KNNDatabase.cuh"
#include <map>

namespace OpenSteer
{
class VehicleGroup : public BaseGroup
{
	typedef std::map< id_type, size_t > IDToIndexMap;

protected:
	int		GetVehicleIndex( id_type _id ) const;

	void	RebuildIDToIndexMap( void );
	/// Copy the host structures to the device.
	void	SyncDevice( void );
	/// Copy the device structures to the host.
	void	SyncHost( void );

	// Vehicle data.
	VehicleGroupData			m_vehicleGroupData;
	VehicleGroupConst			m_vehicleGroupConst;

	IDToIndexMap				m_cIDToIndexMap;

	size_t						m_nCount;

public:
	//VehicleGroup( void );
	VehicleGroup( uint3 const& worldCells, uint const knn );
	virtual ~VehicleGroup( void );

	bool AddVehicle( VehicleData const& vd, VehicleConst const& vc );
	void RemoveVehicle( id_type const id );
	/// Clear all vehicles from the group.
	void Clear( void );
	void SetSyncHost( void )
	{
		m_vehicleGroupData.m_bSyncHost = true;
		m_neighborDB.m_bSyncHost = true;
	}

	VehicleGroupConst &		GetVehicleGroupConst( void )	{ return m_vehicleGroupConst; }
	VehicleGroupData &		GetVehicleGroupData( void )		{ return m_vehicleGroupData; }

	/// Use to extract data for an individual vehicle
	bool GetDataForVehicle( id_type const id, VehicleData &_data, VehicleConst &_const);

	// Overloaded pure virtuals.
	uint		Size( void ) const		{ return m_nCount; }
	float3 *	pdPosition( void )		{ return m_vehicleGroupData.pdPosition(); }

};	//class VehicleGroup
}	//namespace OpenSteer
#endif
