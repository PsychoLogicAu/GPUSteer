#ifndef OPENSTEER_KNNBRUTEFORCECUDA_CUH
#define OPENSTEER_KNNBRUTEFORCECUDA_CUH

#include "AbstractCUDAKernel.h"

namespace OpenSteer
{
class KNNBruteForceCUDA : public AbstractCUDAKernel
{
protected:
	size_t		m_k;

	// Temporary device storage.
	float *		m_pdDistanceMatrix;
	size_t *	m_pdIndexMatrix;

public:
	KNNBruteForceCUDA( VehicleGroup * pVehicleGroup, size_t const k );
	~KNNBruteForceCUDA( void ) {}

	virtual void init( void );
	virtual void run( void );
	virtual void close( void );

};	// class KNNBruteForceCUDA
}	// namespace OpenSteer

#endif