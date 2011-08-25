#ifndef OPENSTEER_STEERTOAVOIDNEIGHBORSCUDA_H
#define OPENSTEER_STEERTOAVOIDNEIGHBORSCUDA_H

#include "AbstractCUDAKernel.h"

namespace OpenSteer
{

class SteerToAvoidNeighborsCUDA : public AbstractCUDAKernel
{
protected:
	float m_fMinTimeToCollision;

public:
	SteerToAvoidNeighborsCUDA( VehicleGroup *pVehicleGroup, float const fMinTimeToCollision );
	~SteerToAvoidNeighborsCUDA( void ) {}

	virtual void init( void );
	virtual void run( void );
	virtual void close(void );
};	// class SteerToAvoidNeighborsCUDA

} // namespace OpenSteer
#endif