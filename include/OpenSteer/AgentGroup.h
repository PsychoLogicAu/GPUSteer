#ifndef OPENSTEER_VEHICLEGROUP_H
#define OPENSTEER_VEHICLEGROUP_H

#include "BaseGroup.h"

#include "AgentGroupData.cuh"

#include <map>

namespace OpenSteer
{
class AgentGroup : public BaseGroup
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
	AgentGroupData				m_agentGroupData;
	AgentGroupConst				m_agentGroupConst;

	IDToIndexMap				m_cIDToIndexMap;

public:
	//AgentGroup( void );
	AgentGroup( uint3 const& worldCells, uint const knn );
	virtual ~AgentGroup( void );

	bool AddVehicle( VehicleData const& vd, VehicleConst const& vc );
	void RemoveVehicle( id_type const id );
	/// Clear all vehicles from the group.
	void Clear( void );
	void SetSyncHost( void )
	{
		m_agentGroupData.m_bSyncHost = true;
		m_neighborDB.m_bSyncHost = true;
	}

	AgentGroupConst &		GetAgentGroupConst( void )	{ return m_agentGroupConst; }
	AgentGroupData &		GetAgentGroupData( void )		{ return m_agentGroupData; }

	/// Use to extract data for an individual vehicle
	bool GetDataForVehicle( id_type const id, VehicleData &_data, VehicleConst &_const);

	// Overloaded pure virtuals.
	virtual float3 *	pdPosition( void )					{ return m_agentGroupData.pdPosition(); }
	virtual float3 *	pdDirection( void )					{ return m_agentGroupData.pdForward(); }
	virtual float *		pdSpeed( void )						{ return m_agentGroupData.pdSpeed(); }
	virtual float *		pdRadius( void )					{ return m_agentGroupConst.pdRadius(); }

};	//class AgentGroup
}	//namespace OpenSteer
#endif
