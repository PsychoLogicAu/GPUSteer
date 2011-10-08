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
	int		GetAgentIndex( id_type _id ) const;

	void	RebuildIDToIndexMap( void );
	/// Copy the host structures to the device.
	void	SyncDevice( void );
	/// Copy the device structures to the host.
	void	SyncHost( void );

	// Vehicle data.
	AgentGroupData				m_agentGroupData;

	IDToIndexMap				m_cIDToIndexMap;

public:
	//AgentGroup( void );
	AgentGroup( uint3 const& worldCells, uint const knn );
	virtual ~AgentGroup( void );

	bool AddAgent( AgentData const& ad );
	void RemoveAgent( id_type const id );
	/// Clear all vehicles from the group.
	void Clear( void );
	virtual void SetSyncHost( void )
	{
		m_agentGroupData.m_bSyncHost = true;
		m_neighborDB.m_bSyncHost = true;
	}

	AgentGroupData &		GetAgentGroupData( void )		{ return m_agentGroupData; }

	/// Use to extract data for an individual vehicle
	bool GetDataForAgent( id_type const id, AgentData & ad );

	// Overloaded pure virtuals.
	virtual float4 *	pdPosition( void )					{ return m_agentGroupData.pdPosition(); }
	virtual float4 *	pdDirection( void )					{ return m_agentGroupData.pdDirection(); }
	virtual float *		pdSpeed( void )						{ return m_agentGroupData.pdSpeed(); }
	virtual float *		pdRadius( void )					{ return m_agentGroupData.pdRadius(); }

};	//class AgentGroup
}	//namespace OpenSteer
#endif
