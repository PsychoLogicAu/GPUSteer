#ifndef OPENSTEER_NEARESTNEIGHBORDATA_CUH
#define OPENSTEER_NEARESTNEIGHBORDATA_CUH

#include "CUDAGlobals.cuh"

#include "dev_vector.cuh"

#include <vector>

namespace OpenSteer
{

class nearest_neighbor_data
{
private:
	// Number of nearest neighbors per agent.
	uint				m_nK;
	// Number of agents.
	size_t				m_nSize;

	// Device vectors
	dev_vector<uint>	m_dvKNNIndices;		// Index of agent.
	dev_vector<float>	m_dvKNNDistances;	// Distance to agent.	// TODO: is this used anywhere?
	dev_vector<float3>	m_dvKNNPositions;
	dev_vector<float3>	m_dvKNNDirections;	// Direction of agent.
	dev_vector<float>	m_dvKNNSpeeds;		// Speed of agent.

	// Host vectors
	std::vector<uint>	m_hvKNNIndices;
	std::vector<float>	m_hvKNNDistances;	// TODO: is this used anywhere?
	std::vector<float3>	m_hvKNNPositions;
	std::vector<float3>	m_hvKNNDirections;	// Direction of agent.
	std::vector<float>	m_hvKNNSpeeds;		// Speed of agent.

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

	// Accessor/mutators
	bool	seedable( void ) const								{ return m_bSeedable; }
	void	seedable( bool const b )							{ m_bSeedable = b; }
	uint	k( void ) const										{ return m_nK; }


	// Device data.
	uint *		pdKNNIndices( void )							{ return m_dvKNNIndices.begin(); }
	float *		pdKNNDistances( void )							{ return m_dvKNNDistances.begin(); }
	float3 *	pdKNNPositions( void )							{ return m_dvKNNPositions.begin(); }
	float3 *	pdKNNDirections( void )							{ return m_dvKNNDirections.begin(); }
	float *		pdKNNSpeeds( void )								{ return m_dvKNNSpeeds.begin(); }

	// Host data.
	std::vector<uint> const& hvKNNIndices( void ) const			{ return m_hvKNNIndices; }
	std::vector<uint> & hvKNNIndices( void )					{ m_bSyncDevice = true; return m_hvKNNIndices; }
	std::vector<float> const& hvKNNDistances( void ) const		{ return m_hvKNNDistances; }
	std::vector<float> & hvKNNDistances( void )					{ m_bSyncDevice = true; return m_hvKNNDistances; }
	std::vector<float3> const& hvKNNPositions( void ) const		{ return m_hvKNNPositions; }
	std::vector<float3> & hvKNNPositions( void )				{ m_bSyncDevice = true; return m_hvKNNPositions; }
	std::vector<float3> const& hvKNNDirections( void ) const	{ return m_hvKNNDirections; }
	std::vector<float3> & hvKNNDirections( void )				{ m_bSyncDevice = true; return m_hvKNNDirections; }
	std::vector<float> const& hvKNNSpeeds( void ) const			{ return m_hvKNNSpeeds; }
	std::vector<float> & hvKNNSpeeds( void )					{ m_bSyncDevice = true; return m_hvKNNSpeeds; }

	/// Adds an agent.
	void addAgent( void )
	{
		for( uint i = 0; i < m_nK; i++ )
		{
			m_hvKNNIndices.push_back( UINT_MAX );
			m_hvKNNDistances.push_back( FLT_MAX );
			m_hvKNNPositions.push_back( make_float3( 0.f, 0.f, 0.f ) );
			m_hvKNNDirections.push_back( make_float3( 0.f, 0.f, 0.f ) );
			m_hvKNNSpeeds.push_back( 0.f );
		}

		m_nSize++;
		m_bSeedable = false;
		m_bSyncDevice = true;
	}

	/// Removes an agent.
	void removeAgent( size_t const index )
	{
		if( index < m_nSize )
		{
			m_hvKNNIndices.erase( m_hvKNNIndices.begin() + (index * m_nK), m_hvKNNIndices.begin() + (index * m_nK) + m_nK );
			m_hvKNNDistances.erase( m_hvKNNDistances.begin() + (index * m_nK), m_hvKNNDistances.begin() + (index * m_nK) + m_nK );
			m_hvKNNPositions.erase( m_hvKNNPositions.begin() + (index * m_nK), m_hvKNNPositions.begin() + (index * m_nK) + m_nK );
			m_hvKNNDirections.erase( m_hvKNNDirections.begin() + (index * m_nK), m_hvKNNDirections.begin() + (index * m_nK) + m_nK );
			m_hvKNNSpeeds.erase( m_hvKNNSpeeds.begin() + (index * m_nK), m_hvKNNSpeeds.begin() + (index * m_nK) + m_nK );

			m_nSize--;
			//m_bSeedable = false;	// Should still be seedable. TODO: ???
			m_bSyncDevice = true;
		}
	}

	/// Get the KNN indices for the agent at index.
	void getAgentData( size_t const index, uint * pKNNIndices, float * pKNNDistances, float3 * pKNNPositions, float3 * pKNNDirections, float * pKNNSpeeds )
	{
		if( index < m_nSize )
		{
			syncHost();

			memcpy( pKNNIndices, &m_hvKNNIndices[ index * m_nK ], m_nK * sizeof(uint) );
			memcpy( pKNNDistances, &m_hvKNNDistances[ index * m_nK ], m_nK * sizeof(float) );
			memcpy( pKNNPositions, &m_hvKNNPositions[ index * m_nK ], m_nK * sizeof(float3) );
			memcpy( pKNNDirections, &m_hvKNNDirections[ index * m_nK ], m_nK * sizeof(float3) );
			memcpy( pKNNSpeeds, &m_hvKNNSpeeds[ index * m_nK ], m_nK * sizeof(float) );
		}
	}

	void syncHost( void )
	{
		if( m_bSyncHost )
		{
			cudaMemcpy( &m_hvKNNIndices[0], pdKNNIndices(), m_nSize * m_nK * sizeof(uint), cudaMemcpyDeviceToHost );
			cudaMemcpy( &m_hvKNNDistances[0], pdKNNDistances(), m_nSize * m_nK * sizeof(float), cudaMemcpyDeviceToHost );
			cudaMemcpy( &m_hvKNNPositions[0], pdKNNPositions(), m_nSize * m_nK * sizeof(float3), cudaMemcpyDeviceToHost );
			cudaMemcpy( &m_hvKNNDirections[0], pdKNNDirections(), m_nSize * m_nK * sizeof(float3), cudaMemcpyDeviceToHost );
			cudaMemcpy( &m_hvKNNSpeeds[0], pdKNNSpeeds(), m_nSize * m_nK * sizeof(float), cudaMemcpyDeviceToHost );

			m_bSyncHost = false;
		}
	}

	void syncDevice( void )
	{
		if( m_bSyncDevice )
		{
			m_dvKNNIndices = m_hvKNNIndices;
			m_dvKNNDistances = m_hvKNNDistances;
			m_dvKNNPositions = m_hvKNNPositions;
			m_dvKNNDirections = m_hvKNNDirections;
			m_dvKNNSpeeds = m_hvKNNSpeeds;

			m_bSyncDevice = false;
		}
	}

	void clear( void )
	{
		m_nSize = 0;
		m_bSyncHost = false;
		m_bSyncDevice = false;
		m_bSeedable = false;

		m_dvKNNIndices.clear();
		m_dvKNNDistances.clear();
		m_dvKNNPositions.clear();
		m_dvKNNDirections.clear();
		m_dvKNNSpeeds.clear();

		m_hvKNNIndices.clear();
		m_hvKNNDistances.clear();
		m_hvKNNPositions.clear();
		m_hvKNNDirections.clear();
		m_hvKNNSpeeds.clear();
	}
};	// class nearest_neighbor_data
typedef nearest_neighbor_data NearestNeighborData;
}	// namespace OpenSteer
#endif
