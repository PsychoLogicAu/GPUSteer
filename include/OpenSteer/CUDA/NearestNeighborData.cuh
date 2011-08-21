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

	// Device vector
	dev_vector<uint>	m_dvKNNIndices;
	dev_vector<float>	m_dvKNNDistances;

	// Host vector
	std::vector<uint>	m_hvKNNIndices;
	std::vector<float>	m_hvKNNDistances;

	bool	m_bSyncHost;
	bool	m_bSyncDevice;

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
	bool	seedable( void ) const							{ return m_bSeedable; }
	void	seedable( bool const b )						{ m_bSeedable = b; }
	uint	k( void ) const									{ return m_nK; }


	// Device data.
	uint *	pdKNNIndices( void )							{ return m_dvKNNIndices.begin(); }
	float *	pdKNNDistances( void )							{ return m_dvKNNDistances.begin(); }

	// Host data.
	std::vector<uint> const& hvKNNIndices( void ) const		{ return m_hvKNNIndices; }
	std::vector<uint> & hvKNNIndices( void )				{ m_bSyncDevice = true; return m_hvKNNIndices; }
	std::vector<float> const& hvKNNDistances( void ) const	{ return m_hvKNNDistances; }
	std::vector<float> & hvKNNDistances( void )				{ m_bSyncDevice = true; return m_hvKNNDistances; }

	/// Adds an agent.
	void addAgent( void )
	{
		for( uint i = 0; i < m_nK; i++ )
		{
			m_hvKNNIndices.push_back( UINT_MAX );
			m_hvKNNDistances.push_back( FLT_MAX );
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

			m_nSize--;
			//m_bSeedable = false;	// Should still be seedable.
			m_bSyncDevice = true;
		}
	}

	/// Get the KNN indices for the agent at index.
	void getAgentData( size_t const index, uint * pKNNIndices, float * pKNNDistances )
	{
		if( index < m_nSize )
		{
			syncHost();

			memcpy( pKNNIndices, &m_hvKNNIndices[ index * m_nK ], m_nK * sizeof(uint) );
			memcpy( pKNNDistances, &m_hvKNNDistances[ index * m_nK ], m_nK * sizeof(float) );
		}
	}

	void syncHost( void )
	{
		if( m_bSyncHost )
		{
			cudaMemcpy( &m_hvKNNIndices[0], pdKNNIndices(), m_nSize * m_nK * sizeof(uint), cudaMemcpyDeviceToHost );
			cudaMemcpy( &m_hvKNNDistances[0], pdKNNDistances(), m_nSize * m_nK * sizeof(float), cudaMemcpyDeviceToHost );

			m_bSyncHost = false;
		}
	}

	void syncDevice( void )
	{
		if( m_bSyncDevice )
		{
			m_dvKNNIndices = m_hvKNNIndices;
			m_dvKNNDistances = m_hvKNNDistances;

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

		m_hvKNNIndices.clear();
		m_hvKNNDistances.clear();
	}
};	// class nearest_neighbor_data
typedef nearest_neighbor_data NearestNeighborData;
}	// namespace OpenSteer
#endif
