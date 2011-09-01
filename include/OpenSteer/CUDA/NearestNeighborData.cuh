#ifndef OPENSTEER_NEARESTNEIGHBORDATA_CUH
#define OPENSTEER_NEARESTNEIGHBORDATA_CUH

#include "CUDAGlobals.cuh"

#include "dev_vector.cuh"

#include <vector>

namespace OpenSteer
{

class nearest_neighbor_data
{
	friend class VehicleGroup;
private:
	// Number of nearest neighbors per agent.
	uint				m_nK;
	// Number of agents.
	size_t				m_nSize;

	//
	// Device vectors
	//
	dev_vector<uint>	m_dvKNNIndices;			// Indices of K Nearest Neighbors.
	dev_vector<float>	m_dvKNNDistances;		// Distances to K Nearest Neighbors.

	dev_vector<uint>	m_dvCellIndices;		// Index of agent's current cell, sorted by agent index.

	dev_vector<uint>	m_dvCellIndicesSorted;	// Index of agent's current cell, sorted by cell index.
	dev_vector<uint>	m_dvAgentIndicesSorted;	// Index of agent, sorted by cell index.

	dev_vector<float3>	m_dvPositionSorted;		// Position of agent, sorted by cell index.
	dev_vector<float3>	m_dvDirectionSorted;	// Direction of agent, sorted by cell index.
	dev_vector<float>	m_dvSpeedSorted;		// Speed of agent, sorted by cell index.

	//
	// Host vectors
	//
	std::vector<uint>	m_hvKNNIndices;
	std::vector<float>	m_hvKNNDistances;

	std::vector<uint>	m_hvCellIndices;

	std::vector<uint>	m_hvCellIndicesSorted;
	std::vector<uint>	m_hvAgentIndicesSorted;

	std::vector<float3>	m_hvPositionSorted;
	std::vector<float3>	m_hvDirectionSorted;	// Direction of agent.
	std::vector<float>	m_hvSpeedSorted;		// Speed of agent.

	bool	m_bSyncHost;
	bool	m_bSyncDevice;

	// Used by KNNBruteForceV3
	bool	m_bSeedable;

public:
	nearest_neighbor_data( uint const k )
		:	m_nSize( 0 ),
			m_nK( k ),
			m_bSyncHost( false ),
			m_bSyncDevice( false ),
			m_bSeedable( false )
	{ }

	~nearest_neighbor_data( void )
	{ }

	//
	// Accessor/mutators
	//
	bool	seedable( void ) const								{ return m_bSeedable; }
	void	seedable( bool const b )							{ m_bSeedable = b; }
	uint	k( void ) const										{ return m_nK; }


	// Device data.
	uint *		pdKNNIndices( void )							{ return m_dvKNNIndices.begin(); }
	float *		pdKNNDistances( void )							{ return m_dvKNNDistances.begin(); }
	
	uint *		pdCellIndices( void )							{ return m_dvCellIndices.begin(); }

	uint *		pdCellIndicesSorted( void )						{ return m_dvCellIndicesSorted.begin(); }
	uint *		pdAgentIndicesSorted( void )					{ return m_dvAgentIndicesSorted.begin(); }

	float3 *	pdPositionSorted( void )						{ return m_dvPositionSorted.begin(); }
	float3 *	pdDirectionSorted( void )						{ return m_dvDirectionSorted.begin(); }
	float *		pdSpeedSorted( void )							{ return m_dvSpeedSorted.begin(); }

	// Host data.
	std::vector<uint> const& hvKNNIndices( void ) const			{ return m_hvKNNIndices; }
	std::vector<uint> & hvKNNIndices( void )					{ m_bSyncDevice = true; return m_hvKNNIndices; }
	std::vector<float> const& hvKNNDistances( void ) const		{ return m_hvKNNDistances; }
	std::vector<float> & hvKNNDistances( void )					{ m_bSyncDevice = true; return m_hvKNNDistances; }

	std::vector<uint> const& hvCellIndices( void ) const		{ return m_hvCellIndices; }
	std::vector<uint> & hvCellIndices( void )					{ m_bSyncDevice = true; return m_hvCellIndices; }
	
	std::vector<uint> const& hvCellIndicesSorted( void ) const	{ return m_hvCellIndicesSorted; }
	std::vector<uint> & hvCellIndicesSorted( void )				{ m_bSyncDevice = true; return m_hvCellIndicesSorted; }
	std::vector<uint> const& hvAgentIndicesSorted( void ) const	{ return m_hvAgentIndicesSorted; }
	std::vector<uint> & hvAgentIndicesSorted( void )			{ m_bSyncDevice = true; return m_hvAgentIndicesSorted; }

	std::vector<float3> const& hvPositionSorted( void ) const	{ return m_hvPositionSorted; }
	std::vector<float3> & hvPositionSorted( void )				{ m_bSyncDevice = true; return m_hvPositionSorted; }
	std::vector<float3> const& hvDirectionSorted( void ) const	{ return m_hvDirectionSorted; }
	std::vector<float3> & hvDirectionSorted( void )				{ m_bSyncDevice = true; return m_hvDirectionSorted; }
	std::vector<float> const& hvSpeedSorted( void ) const		{ return m_hvSpeedSorted; }
	std::vector<float> & hvSpeedSorted( void )					{ m_bSyncDevice = true; return m_hvSpeedSorted; }

	/// Adds an agent.
	void addAgent( void );

	/// Removes an agent.
	void removeAgent( size_t const index );

	/// Get the KNN data for the agent at index.
	void getAgentData( size_t const index, uint * pKNNIndices, float * pKNNDistances, uint & cellIndex );

	void syncHost( void );

	void syncDevice( void );

	void clear( void );
};	// class nearest_neighbor_data
typedef nearest_neighbor_data NearestNeighborData;
}	// namespace OpenSteer
#endif
