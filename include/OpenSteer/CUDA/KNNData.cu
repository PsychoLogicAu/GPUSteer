#include "KNNData.cuh"

using namespace OpenSteer;

KNNData::KNNData( uint const nSize, uint const nK )
:	m_bSeedable( false ),
	m_bSyncHost( false )
{
	resize( nSize, nK );
}

void KNNData::syncHost( void )
{
	if( m_bSyncHost )
	{
		CUDA_SAFE_CALL( cudaMemcpy( &m_hvKNNIndices[0], pdKNNIndices(), m_nSize * m_nK * sizeof(uint), cudaMemcpyDeviceToHost ) );
		CUDA_SAFE_CALL( cudaMemcpy( &m_hvKNNDistances[0], pdKNNDistances(), m_nSize * m_nK * sizeof(float), cudaMemcpyDeviceToHost ) );

		m_bSyncHost = false;
	}
}

void KNNData::resize( uint const nSize, uint const nK )
{
	m_nSize = nSize;
	m_nK = nK;

	// Resize the host data.
	m_hvKNNIndices.resize( nSize * nK );
	m_hvKNNDistances.resize( nSize * nK );

	// Resize the device data.
	m_dvKNNIndices.resize( nSize * nK );
	m_dvKNNDistances.resize( nSize * nK );

	m_bSeedable = false;
}

void KNNData::clear( void )
{
	m_nSize = 0;
	m_bSyncHost = false;

	// Clear the device data.
	m_dvKNNDistances.clear();
	m_dvKNNIndices.clear();

	// Clear the host data.
	m_hvKNNDistances.clear();
	m_hvKNNIndices.clear();

	m_bSeedable = false;
}

void KNNData::getAgentData( size_t const index, uint * pKNNIndices, float * pKNNDistances )
{
	if( index < m_nSize )
	{
		// Pull the data from the GPU.
		syncHost();

		// Copy into the parameters.
		memcpy( pKNNIndices, &m_hvKNNIndices[ index * m_nK ], m_nK * sizeof(uint) );
		memcpy( pKNNDistances, &m_hvKNNDistances[ index * m_nK ], m_nK * sizeof(float) );
	}
}
