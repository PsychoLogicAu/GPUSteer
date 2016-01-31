#ifndef OPENSTEER_STEERTOAVOIDNEIGHBORSCUDA_H
#define OPENSTEER_STEERTOAVOIDNEIGHBORSCUDA_H

#include "AbstractCUDAKernel.cuh"

#include "KNNData.cuh"

namespace OpenSteer
{

class SteerToAvoidNeighborsCUDA : public AbstractCUDAKernel
{
protected:
	bool			m_bAvoidCloseNeighbors;
	float			m_fMinTimeToCollision;
	float			m_fMinSeparationDistance;

	KNNData *		m_pKNNData;
	AgentGroup *	m_pOtherGroup;

public:
	SteerToAvoidNeighborsCUDA( AgentGroup * pAgentGroup, KNNData * pKNNData, AgentGroup * pOtherGroup, float const fMinTimeToCollision, float const fMinSeparationDistance, bool const bAvoidCloseNeighbors, float const fWeight, uint const doNotApplyWith );
	virtual ~SteerToAvoidNeighborsCUDA( void ) {}

	virtual void init( void );
	virtual void run( void );
	virtual void close(void );
};	// class SteerToAvoidNeighborsCUDA

} // namespace OpenSteer
#endif