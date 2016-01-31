#ifndef OPENSTEER_ANTIPENETRATIONAGENTSCUDA_CUH
#define OPENSTEER_ANTIPENETRATIONAGENTSCUDA_CUH

#include "AbstractCUDAKernel.cuh"

#include "KNNData.cuh"

namespace OpenSteer
{
class AntiPenetrationAgentsCUDA : public AbstractCUDAKernel
{
protected:
	KNNData *		m_pKNNData;
	AgentGroup *	m_pOtherGroup;

	float4 *		m_pdPositionNew;


public:
	AntiPenetrationAgentsCUDA( AgentGroup * pAgentGroup, KNNData * pKNNData, AgentGroup * pOtherGroup, uint const doNotApplyWith );
	virtual ~AntiPenetrationAgentsCUDA( void ) {}

	virtual void init( void );
	virtual void run( void );
	virtual void close( void );
};	// class AntiPenetrationAgentsCUDA
}	// namespace OpenSteer
#endif