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

	//
	// Device vectors
	//
	dev_vector<uint>	m_dvKNNIndices;			// Indices of K Nearest Neighbors.
	dev_vector<float>	m_dvKNNDistances;		// Distances to K Nearest Neighbors.

	dev_vector<uint>	m_dvCellIndices;
	dev_vector<uint>	m_dvAgentIndices;

	dev_vector<float3>	m_dvPositionSorted;		
	dev_vector<float3>	m_dvDirectionSorted;	// Direction of agent.
	dev_vector<float>	m_dvSpeedSorted;		// Speed of agent.

	//
	// Host vectors
	//
	std::vector<uint>	m_hvKNNIndices;
	std::vector<float>	m_hvKNNDistances;

	std::vector<uint>	m_hvCellIndices;
	std::vector<uint>	m_hvAgentIndices;

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
	uint *		pdAgentIndices( void )							{ return m_dvAgentIndices.begin(); }

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
	std::vector<uint> const& hvAgentIndices( void ) const		{ return m_hvAgentIndices; }
	std::vector<uint> & hvAgentIndices( void )					{ m_bSyncDevice = true; return m_hvAgentIndices; }

	std::vector<float3> const& hvPositionSorted( void ) const	{ return m_hvPositionSorted; }
	std::vector<float3> & hvPositionSorted( void )				{ m_bSyncDevice = true; return m_hvPositionSorted; }
	std::vector<float3> const& hvDirectionSorted( void ) const	{ return m_hvDirectionSorted; }
	std::vector<float3> & hvDirectionSorted( void )				{ m_bSyncDevice = true; return m_hvDirectionSorted; }
	std::vector<float> const& hvSpeedSorted( void ) const		{ return m_hvSpeedSorted; }
	std::vector<float> & hvSpeedSorted( void )					{ m_bSyncDevice = true; return m_hvSpeedSorted; }

	/// Adds an agent.
	void addAgent( void )
	{
		for( uint i = 0; i < m_nK; i++ )
		{
			m_hvKNNIndices.push_back( UINT_MAX );
			m_hvKNNDistances.push_back( FLT_MAX );
		}

		m_hvCellIndices.push_back( UINT_MAX );
		m_hvAgentIndices.push_back( UINT_MAX );

		m_hvPositionSorted.push_back( make_float3( 0.f, 0.f, 0.f ) );
		m_hvDirectionSorted.push_back( make_float3( 0.f, 0.f, 0.f ) );
		m_hvSpeedSorted.push_back( 0.f );

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

			// These won't be the right agents... but does it even matter?
			// We won't ever try to add/remove them _during_ simulation.
			// TODO: If any changes made, force recalculate of KNN kernel on next tick.
			m_hvCellIndices.erase( m_hvCellIndices.begin() + index );
			m_hvAgentIndices.erase( m_hvAgentIndices.begin() + index );

			m_hvPositionSorted.erase( m_hvPositionSorted.begin() + index );
			m_hvDirectionSorted.erase( m_hvDirectionSorted.begin() + index );
			m_hvSpeedSorted.erase( m_hvSpeedSorted.begin() + index );

			m_nSize--;
			//m_bSeedable = false;	// Should still be seedable. TODO: check that hypothesis???
			m_bSyncDevice = true;
		}
	}

	/*
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
	*/

	void syncHost( void )
	{
		if( m_bSyncHost )
		{
			CUDA_SAFE_CALL( cudaMemcpy( &m_hvKNNIndices[0], pdKNNIndices(), m_nSize * m_nK * sizeof(uint), cudaMemcpyDeviceToHost ) );
			CUDA_SAFE_CALL( cudaMemcpy( &m_hvKNNDistances[0], pdKNNDistances(), m_nSize * m_nK * sizeof(float), cudaMemcpyDeviceToHost ) );

			CUDA_SAFE_CALL( cudaMemcpy( &m_hvAgentIndices[0], pdAgentIndices(), m_nSize * sizeof(uint), cudaMemcpyDeviceToHost ) );
			CUDA_SAFE_CALL( cudaMemcpy( &m_hvCellIndices[0], pdCellIndices(), m_nSize * sizeof(uint), cudaMemcpyDeviceToHost ) );

			CUDA_SAFE_CALL( cudaMemcpy( &m_hvPositionSorted[0], pdPositionSorted(), m_nSize * sizeof(float3), cudaMemcpyDeviceToHost ) );
			CUDA_SAFE_CALL( cudaMemcpy( &m_hvDirectionSorted[0], pdDirectionSorted(), m_nSize * sizeof(float3), cudaMemcpyDeviceToHost ) );
			CUDA_SAFE_CALL( cudaMemcpy( &m_hvSpeedSorted[0], pdSpeedSorted(), m_nSize * sizeof(float), cudaMemcpyDeviceToHost ) );

			m_bSyncHost = false;
		}
	}

	void syncDevice( void )
	{
		if( m_bSyncDevice )
		{
			m_dvKNNIndices = m_hvKNNIndices;
			m_dvKNNDistances = m_hvKNNDistances;

			m_dvCellIndices = m_hvCellIndices;
			m_dvAgentIndices = m_hvAgentIndices;

			m_dvPositionSorted = m_hvPositionSorted;
			m_dvDirectionSorted = m_hvDirectionSorted;
			m_dvSpeedSorted = m_hvSpeedSorted;

			m_bSyncDevice = false;
		}
	}

	void clear( void )
	{
		m_nSize = 0;
		m_bSyncHost = false;
		m_bSyncDevice = false;
		m_bSeedable = false;

		// Clear device data.
		m_dvKNNIndices.clear();
		m_dvKNNDistances.clear();
		m_dvCellIndices.clear();
		m_dvAgentIndices.clear();
		m_dvPositionSorted.clear();
		m_dvDirectionSorted.clear();
		m_dvSpeedSorted.clear();

		// Clear host data.
		m_hvKNNIndices.clear();
		m_hvKNNDistances.clear();
		m_hvCellIndices.clear();
		m_hvAgentIndices.clear();
		m_hvPositionSorted.clear();
		m_hvDirectionSorted.clear();
		m_hvSpeedSorted.clear();
	}
};	// class nearest_neighbor_data
typedef nearest_neighbor_data NearestNeighborData;
}	// namespace OpenSteer
#endif
