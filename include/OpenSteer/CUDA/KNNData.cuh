#ifndef OPENSTEER_KNNDATA_CUH
#define OPENSTEER_KNNDATA_CUH

#include "dev_vector.cuh"
#include "CUDAGlobals.cuh"
#include <vector>

namespace OpenSteer
{
class KNNData
{
private:
	uint				m_nK;
	uint				m_nSize;
	bool				m_bSyncHost;

	// Used by KNNBruteForceV3
	bool				m_bSeedable;

	// Device vectors.
	dev_vector<uint>	m_dvKNNIndices;			// Indices of K Nearest Neighbors.
	dev_vector<float>	m_dvKNNDistances;		// Distances to K Nearest Neighbors.

	// Host vectors.
	std::vector<uint>	m_hvKNNIndices;
	std::vector<float>	m_hvKNNDistances;

	void syncHost( void );

public:
	KNNData( uint const nSize, uint const nK );
	~KNNData( void );

	//
	// Accessor/mutators
	//
	uint *		pdKNNIndices( void )							{ return m_dvKNNIndices.begin(); }
	float *		pdKNNDistances( void )							{ return m_dvKNNDistances.begin(); }

	std::vector<uint> const& hvKNNIndices( void ) const			{ return m_hvKNNIndices; }
	std::vector<float> const& hvKNNDistances( void ) const		{ return m_hvKNNDistances; }

	uint		k( void )										{ return m_nK; }
	uint		size( void )									{ return m_nSize; }

	void resize( uint const nSize, uint const nK );

	void clear( void );

	/// Get the KNN data for the agent at index.
	void getAgentData( uint const index, uint * pKNNIndices, float * pKNNDistances );
};	// class KNNData
};	// namespace OpenSteer

#endif
