#ifndef OPENSTEER_VEHICLEGROUP_H
#define OPENSTEER_VEHICLEGROUP_H

#include "VehicleGroupData.cuh"
#include "CUDA/VehicleGroupBinData.cuh"
#include "CUDA/NearestNeighborData.cuh"
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

	// Vehicle data.
	VehicleGroupData			m_vehicleGroupData;
	VehicleGroupConst			m_vehicleGroupConst;

	// Nearest neighbor data.
	NearestNeighborData			m_nearestNeighbors;
	NearestNeighborData			m_nearestObstacles;

	// Bin data to be used for KNN lookups.
	BinData						m_binData;

	IDToIndexMap				m_cIDToIndexMap;

	size_t						m_nCount;

public:
	//VehicleGroup( void );
	VehicleGroup( uint3 const& worldCells, float3 const& worldSize );
	virtual ~VehicleGroup( void );

	bool AddVehicle( VehicleData const& vd, VehicleConst const& vc );
	void RemoveVehicle( id_type const id );
	/// Clear all vehicles from the group.
	void Clear( void );
	void SetSyncHost( void )
	{
		m_vehicleGroupData.m_bSyncHost = true;
	}

	/// Get the size of the collection.
	size_t Size( void ) const
	{
		return m_nCount;
	}

	NearestNeighborData &	GetNearestNeighborData( void )	{ return m_nearestNeighbors; }
	NearestNeighborData &	GetNearestObstacleData( void )	{ return m_nearestNeighbors; }
	VehicleGroupConst &		GetVehicleGroupConst( void )	{ return m_vehicleGroupConst; }
	VehicleGroupData &		GetVehicleGroupData( void )		{ return m_vehicleGroupData; }
	BinData &				GetBinData( void )				{ return m_binData; }
	
	/// Use to extract data for an individual vehicle
	bool GetDataForVehicle( id_type const id, VehicleData &_data, VehicleConst &_const);

	//void OutputDataToFile( const char *filename );
};	//class VehicleGroup
}	//namespace OpenSteer
#endif
