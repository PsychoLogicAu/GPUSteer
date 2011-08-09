#ifndef OPENSTEER_KNNBINNING_CUH
#define OPENSTEER_KNNBINNING_CUH

#include "AbstractCUDAKernel.h"

#include "VehicleGroupBinData.cuh"

namespace OpenSteer
{
class KNNBinningCUDA : public AbstractCUDAKernel
{
protected:
	size_t			m_k;					// Number of nearest neighbors to search for.

	// The following are used as key/value to sort vehicles into bins.
	size_t *		m_pdAgentCellIndices;	// Key: the bin this vehicle is in.
	size_t *		m_pdAgentIndices;		// Value: the index of this vehicle in the group.

	// Passed in from externally.
	bin_data *		m_pdBinData;			// Device bin data.


public:
	KNNBinningCUDA( VehicleGroup * pVehicleGroup, size_t const k );
	virtual ~KNNBinningCUDA( void ) {}

	virtual void init( void );
	virtual void run( void );
	virtual void close( void );




};	// class KNNBinningCUDA
}	// namespace OpenSteer


#endif