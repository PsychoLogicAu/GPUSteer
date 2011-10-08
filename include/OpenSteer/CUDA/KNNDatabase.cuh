#ifndef OPENSTEER_NEARESTNEIGHBORDATA_CUH
#define OPENSTEER_NEARESTNEIGHBORDATA_CUH

#include "CUDAGlobals.cuh"

#include "dev_vector.cuh"

#include <vector>

namespace OpenSteer
{

class KNNDatabase
{
	friend class AgentGroup;
	friend class ObstacleGroup;
private:
	// Number of nearest neighbors per agent.
	uint				m_nK;
	// Number of agents.
	uint				m_nSize;
	// Number of cells.
	uint				m_nCells;

	//
	// Device vectors
	//
	dev_vector< uint >	m_dvCellStart;
	dev_vector< uint >	m_dvCellEnd;

	dev_vector<uint>	m_dvCellIndices;		// Index of agent's current cell, sorted by agent index.

	dev_vector<uint>	m_dvCellIndicesSorted;	// Index of agent's current cell, sorted by cell index.
	dev_vector<uint>	m_dvAgentIndicesSorted;	// Index of agent, sorted by cell index.

	dev_vector<float4>	m_dvPositionSorted;		// Position of agent, sorted by cell index.

	//
	// Host vectors
	//
	std::vector<uint>	m_hvCellIndices;

	std::vector<uint>	m_hvCellIndicesSorted;
	std::vector<uint>	m_hvAgentIndicesSorted;

	std::vector<float4>	m_hvPositionSorted;

	bool	m_bSyncHost;
	bool	m_bSyncDevice;

public:
	KNNDatabase( uint const k, uint const size, uint const cells )
		:	m_nSize( size ),
			m_nK( k ),
			m_nCells( cells ),
			m_bSyncHost( false ),
			m_bSyncDevice( false )
	{
		resize( size );
		resizeCells( cells );
	}

	~KNNDatabase( void )
	{ }

	//
	// Accessor/mutators
	//
	uint const&	k( void ) const									{ return m_nK; }

	// Device data.
	uint *		pdCellStart( void )								{ return m_dvCellStart.begin(); }
	uint *		pdCellEnd( void )								{ return m_dvCellEnd.begin(); }
	
	uint *		pdCellIndices( void )							{ return m_dvCellIndices.begin(); }

	uint *		pdCellIndicesSorted( void )						{ return m_dvCellIndicesSorted.begin(); }
	uint *		pdAgentIndicesSorted( void )					{ return m_dvAgentIndicesSorted.begin(); }

	float4 *	pdPositionSorted( void )						{ return m_dvPositionSorted.begin(); }

	// Host data.
	std::vector<uint> const& hvCellIndices( void ) const		{ return m_hvCellIndices; }
	std::vector<uint> & hvCellIndices( void )					{ m_bSyncDevice = true; return m_hvCellIndices; }
	
	std::vector<uint> const& hvCellIndicesSorted( void ) const	{ return m_hvCellIndicesSorted; }
	std::vector<uint> & hvCellIndicesSorted( void )				{ m_bSyncDevice = true; return m_hvCellIndicesSorted; }
	std::vector<uint> const& hvAgentIndicesSorted( void ) const	{ return m_hvAgentIndicesSorted; }
	std::vector<uint> & hvAgentIndicesSorted( void )			{ m_bSyncDevice = true; return m_hvAgentIndicesSorted; }

	std::vector<float4> const& hvPositionSorted( void ) const	{ return m_hvPositionSorted; }
	std::vector<float4> & hvPositionSorted( void )				{ m_bSyncDevice = true; return m_hvPositionSorted; }

	void resize( uint const nSize );
	void resizeCells( uint const nSize );

	// TODO: make these private?
	void syncHost( void );
	void syncDevice( void );

	void clear( void );
};	// class KNNDatabase
}	// namespace OpenSteer
#endif
