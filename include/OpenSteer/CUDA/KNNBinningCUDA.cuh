#ifndef OPENSTEER_KNNBINNING_CUH
#define OPENSTEER_KNNBINNING_CUH

#include "AbstractCUDAKernel.h"

#include "KNNBinData.cuh"

namespace OpenSteer
{
class KNNBinningCUDA : public AbstractCUDAKernel
{
protected:
	size_t			m_nCells;

	// The following are used as key/value to sort vehicles into bins.
	// Moved to NearestNeighborData
	//uint *			m_pdCellIndices;	// Key: Index of the cell this agent is in.
	//uint *			m_pdAgentIndices;	// Value: Index of this agent in the group.

	uint *			m_pdCellStart;
	uint *			m_pdCellEnd;

	// Passed in from externally.
	//bin_data *		m_pdBinData;			// Device bin data.

	NearestNeighborData *	m_pNearestNeighborData;


public:
	KNNBinningCUDA( VehicleGroup * pVehicleGroup );
	virtual ~KNNBinningCUDA( void ) {}

	virtual void init( void );
	virtual void run( void );
	virtual void close( void );




};	// class KNNBinningCUDA
}	// namespace OpenSteer


#endif