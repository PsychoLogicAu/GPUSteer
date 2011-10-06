#ifndef OPENSTEER_STEERFORCOHESIONCUDA_H
#define OPENSTEER_STEERFORCOHESIONCUDA_H

#include "AbstractCUDAKernel.cuh"

#include "KNNData.cuh"

namespace OpenSteer
{
class SteerForCohesionCUDA : public AbstractCUDAKernel
{
protected:
	KNNData *		m_pKNNData;
	AgentGroup *	m_pOtherGroup;

	float			m_fMinDistance;
	float			m_fMaxDistance;
	float			m_fCosMaxAngle;

public:
	SteerForCohesionCUDA(	AgentGroup * pAgentGroup, KNNData * pKNNData, AgentGroup * pOtherGroup, float const minDistance, float const maxDistance, float const cosMaxAngle, float const fWeight, uint const doNotApplyWith );
	virtual ~SteerForCohesionCUDA( void ) {}

	virtual void init( void );
	virtual void run( void );
	virtual void close( void );

};	// class SteerForCohesionCUDA
}	// namespace OpenSteer
#endif