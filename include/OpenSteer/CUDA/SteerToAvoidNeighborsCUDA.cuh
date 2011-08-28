#ifndef OPENSTEER_STEERTOAVOIDNEIGHBORSCUDA_H
#define OPENSTEER_STEERTOAVOIDNEIGHBORSCUDA_H

#include "AbstractCUDAKernel.cuh"

namespace OpenSteer
{

class SteerToAvoidNeighborsCUDA : public AbstractCUDAKernel
{
protected:
	float m_fMinTimeToCollision;
	float m_fMinSeparationDistance;

public:
	SteerToAvoidNeighborsCUDA( VehicleGroup *pVehicleGroup, float const fMinTimeToCollision, float const fMinSeparationDistance );
	~SteerToAvoidNeighborsCUDA( void ) {}

	virtual void init( void );
	virtual void run( void );
	virtual void close(void );
};	// class SteerToAvoidNeighborsCUDA

} // namespace OpenSteer
#endif