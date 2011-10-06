#ifndef OPENSTEER_STEERFORSEPARATIONCUDA_H
#define OPENSTEER_STEERFORSEPARATIONCUDA_H

#include "AbstractCUDAKernel.cuh"

#include "KNNData.cuh"

namespace OpenSteer
{
class SteerForSeparationCUDA : public AbstractCUDAKernel
{
protected:
	KNNData *		m_pKNNData;
	AgentGroup *	m_pOtherGroup;

	float			m_fMinDistance;
	float			m_fMaxDistance;
	float			m_fCosMaxAngle;

public:
	SteerForSeparationCUDA(	AgentGroup * pAgentGroup, KNNData * pKNNData, AgentGroup * pOtherGroup, float const minDistance, float const maxDistance, float const cosMaxAngle, float const fWeight, uint const doNotApplyWith );
	virtual ~SteerForSeparationCUDA( void ) {}

	virtual void init( void );
	virtual void run( void );
	virtual void close( void );

};	// class SteerForSeparationCUDA
}	// namespace OpenSteer
#endif