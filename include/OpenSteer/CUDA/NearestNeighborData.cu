#include "NearestNeighborData.cuh"

#include <algorithm>

#include <float.h>

using namespace OpenSteer;

void nearest_neighbor_data::addAgent( void )
{
	for( uint i = 0; i < m_nK; i++ )
	{
		m_hvKNNIndices.push_back( UINT_MAX );
		m_hvKNNDistances.push_back( FLT_MAX );
	}

	m_hvCellIndices.push_back( UINT_MAX );

	m_hvCellIndicesSorted.push_back( UINT_MAX );
	m_hvAgentIndicesSorted.push_back( UINT_MAX );

	m_hvPositionSorted.push_back( make_float3( 0.f, 0.f, 0.f ) );
	m_hvDirectionSorted.push_back( make_float3( 0.f, 0.f, 0.f ) );
	m_hvSpeedSorted.push_back( 0.f );

	m_nSize++;
	m_bSeedable = false;
	m_bSyncDevice = true;
}

void nearest_neighbor_data::removeAgent( size_t const index )
{
	if( index < m_nSize )
	{
		m_hvKNNIndices.erase( m_hvKNNIndices.begin() + (index * m_nK), m_hvKNNIndices.begin() + (index * m_nK) + m_nK );
		m_hvKNNDistances.erase( m_hvKNNDistances.begin() + (index * m_nK), m_hvKNNDistances.begin() + (index * m_nK) + m_nK );

		// TODO: If any changes made, force recalculate of KNN kernel on next tick.
		m_hvCellIndices.erase( m_hvCellIndices.begin() + index );

		m_hvCellIndicesSorted.erase( m_hvCellIndicesSorted.begin() + index );
		m_hvAgentIndicesSorted.erase( m_hvAgentIndicesSorted.begin() + index );

		m_hvPositionSorted.erase( m_hvPositionSorted.begin() + index );
		m_hvDirectionSorted.erase( m_hvDirectionSorted.begin() + index );
		m_hvSpeedSorted.erase( m_hvSpeedSorted.begin() + index );

		m_nSize--;
		m_bSeedable = false;
		m_bSyncDevice = true;
	}
}

void nearest_neighbor_data::getAgentData( size_t const index, uint * pKNNIndices, float * pKNNDistances, uint & cellIndex )
{
	if( index < m_nSize )
	{
		// Pull the data from the GPU.
		syncHost();

		// Copy into the parameters.
		memcpy( pKNNIndices, &m_hvKNNIndices[ index * m_nK ], m_nK * sizeof(uint) );
		memcpy( pKNNDistances, &m_hvKNNDistances[ index * m_nK ], m_nK * sizeof(float) );

		//std::copy( m_hvKNNIndices.begin() + index*m_nK, m_hvKNNIndices.begin() + index*m_nK + m_nK, pKNNIndices );
		//std::copy( m_hvKNNDistances.begin() + index*m_nK, m_hvKNNDistances.begin() + index*m_nK + m_nK, pKNNDistances );

		cellIndex = m_hvCellIndices[ index ];
	}
}

void nearest_neighbor_data::syncHost( void )
{
	if( m_bSyncHost )
	{
		CUDA_SAFE_CALL( cudaMemcpy( &m_hvKNNIndices[0], pdKNNIndices(), m_nSize * m_nK * sizeof(uint), cudaMemcpyDeviceToHost ) );
		CUDA_SAFE_CALL( cudaMemcpy( &m_hvKNNDistances[0], pdKNNDistances(), m_nSize * m_nK * sizeof(float), cudaMemcpyDeviceToHost ) );

		CUDA_SAFE_CALL( cudaMemcpy( &m_hvCellIndices[0], pdCellIndices(), m_nSize * sizeof(uint), cudaMemcpyDeviceToHost ) );

		CUDA_SAFE_CALL( cudaMemcpy( &m_hvCellIndicesSorted[0], pdCellIndicesSorted(), m_nSize * sizeof(uint), cudaMemcpyDeviceToHost ) );
		CUDA_SAFE_CALL( cudaMemcpy( &m_hvAgentIndicesSorted[0], pdAgentIndicesSorted(), m_nSize * sizeof(uint), cudaMemcpyDeviceToHost ) );

		CUDA_SAFE_CALL( cudaMemcpy( &m_hvPositionSorted[0], pdPositionSorted(), m_nSize * sizeof(float3), cudaMemcpyDeviceToHost ) );
		CUDA_SAFE_CALL( cudaMemcpy( &m_hvDirectionSorted[0], pdDirectionSorted(), m_nSize * sizeof(float3), cudaMemcpyDeviceToHost ) );
		CUDA_SAFE_CALL( cudaMemcpy( &m_hvSpeedSorted[0], pdSpeedSorted(), m_nSize * sizeof(float), cudaMemcpyDeviceToHost ) );

		m_bSyncHost = false;
	}
}

void nearest_neighbor_data::syncDevice( void )
{
	if( m_bSyncDevice )
	{
		m_dvKNNIndices = m_hvKNNIndices;
		m_dvKNNDistances = m_hvKNNDistances;

		m_dvCellIndices = m_hvCellIndices;

		m_dvCellIndicesSorted = m_hvCellIndicesSorted;
		m_dvAgentIndicesSorted = m_hvAgentIndicesSorted;

		m_dvPositionSorted = m_hvPositionSorted;
		m_dvDirectionSorted = m_hvDirectionSorted;
		m_dvSpeedSorted = m_hvSpeedSorted;

		m_bSyncDevice = false;
	}
}

void nearest_neighbor_data::clear( void )
{
	m_nSize = 0;
	m_bSyncHost = false;
	m_bSyncDevice = false;
	m_bSeedable = false;

	// Clear device data.
	m_dvKNNIndices.clear();
	m_dvKNNDistances.clear();

	m_dvCellIndices.clear();

	m_dvCellIndicesSorted.clear();
	m_dvAgentIndicesSorted.clear();

	m_dvPositionSorted.clear();
	m_dvDirectionSorted.clear();
	m_dvSpeedSorted.clear();

	// Clear host data.
	m_hvKNNIndices.clear();
	m_hvKNNDistances.clear();

	m_hvCellIndices.clear();

	m_hvCellIndicesSorted.clear();
	m_hvAgentIndicesSorted.clear();

	m_hvPositionSorted.clear();
	m_hvDirectionSorted.clear();
	m_hvSpeedSorted.clear();
}
