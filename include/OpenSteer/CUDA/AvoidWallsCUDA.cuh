#ifndef OPENSTEER_AVOIDWALLSCUDA_H
#define OPENSTEER_AVOIDWALLSCUDA_H

#include "AbstractCUDAKernel.cuh"

#include "KNNData.cuh"
#include "../WallGroup.h"

namespace OpenSteer
{
class AvoidWallsCUDA : public AbstractCUDAKernel
{
protected:
	WallGroup *		m_pWallGroup;
	KNNData *		m_pKNNData;

	float			m_fMinTimeToCollision;

public:
	AvoidWallsCUDA( AgentGroup * pAgentGroup, KNNData * pKNNData, WallGroup * pWallGroup, float const fMinTimeToCollision, float const fWeight, uint const doNotApplyWith );
	virtual ~AvoidWallsCUDA( void ) {}

	virtual void init( void );
	virtual void run( void );
	virtual void close( void );

};	// class AvoidWallsCUDA
}	// namespace OpenSteer
#endif
