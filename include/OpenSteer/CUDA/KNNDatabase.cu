#include "KNNDatabase.cuh"

#include <algorithm>

#include <float.h>

using namespace OpenSteer;

void KNNDatabase::resize( uint const nSize )
{
	m_hvCellIndices.resize( nSize );

	m_hvCellIndicesSorted.resize( nSize );
	m_hvAgentIndicesSorted.resize( nSize );

	m_hvPositionSorted.resize( nSize );

	m_nSize = nSize;
	m_bSyncDevice = true;
}

void KNNDatabase::resizeCells( uint const nCells )
{
	m_dvCellStart.resize( nCells );
	m_dvCellEnd.resize( nCells );

	m_nCells = nCells;
	m_bSyncDevice = true;
}

/*
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

		m_nSize--;
		m_bSeedable = false;
		m_bSyncDevice = true;
	}
}
*/

void KNNDatabase::syncHost( void )
{
	if( m_bSyncHost )
	{
		CUDA_SAFE_CALL( cudaMemcpy( &m_hvCellIndices[0], pdCellIndices(), m_nSize * sizeof(uint), cudaMemcpyDeviceToHost ) );

		CUDA_SAFE_CALL( cudaMemcpy( &m_hvCellIndicesSorted[0], pdCellIndicesSorted(), m_nSize * sizeof(uint), cudaMemcpyDeviceToHost ) );
		CUDA_SAFE_CALL( cudaMemcpy( &m_hvAgentIndicesSorted[0], pdAgentIndicesSorted(), m_nSize * sizeof(uint), cudaMemcpyDeviceToHost ) );

		CUDA_SAFE_CALL( cudaMemcpy( &m_hvPositionSorted[0], pdPositionSorted(), m_nSize * sizeof(float3), cudaMemcpyDeviceToHost ) );

		m_bSyncHost = false;
	}
}

void KNNDatabase::syncDevice( void )
{
	if( m_bSyncDevice )
	{
		m_dvCellIndices = m_hvCellIndices;

		m_dvCellIndicesSorted = m_hvCellIndicesSorted;
		m_dvAgentIndicesSorted = m_hvAgentIndicesSorted;

		m_dvPositionSorted = m_hvPositionSorted;

		m_bSyncDevice = false;
	}
}

void KNNDatabase::clear( void )
{
	m_nSize = 0;
	m_bSyncHost = false;
	m_bSyncDevice = false;

	// Clear device data.
	m_dvCellIndices.clear();

	m_dvCellIndicesSorted.clear();
	m_dvAgentIndicesSorted.clear();

	m_dvPositionSorted.clear();

	// Clear host data.
	m_hvCellIndices.clear();

	m_hvCellIndicesSorted.clear();
	m_hvAgentIndicesSorted.clear();

	m_hvPositionSorted.clear();
}
