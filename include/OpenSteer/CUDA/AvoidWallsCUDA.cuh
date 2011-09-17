#ifndef OPENSTEER_AVOIDWALLSCUDA_H
#define OPENSTEER_AVOIDWALLSCUDA_H

#include "AbstractCUDAKernel.cuh"

namespace OpenSteer
{
class AvoidWallsCUDA : public AbstractCUDAKernel
{
protected:


public:
	AvoidWallsCUDA( AgentGroup * pAgentGroup, float const fWeight );
	virtual ~AvoidWallsCUDA( void ) {}

	virtual void init( void );
	virtual void run( void );
	virtual void close( void );

};	// class AvoidWallsCUDA
};	// namespace OpenSteer


#endif
