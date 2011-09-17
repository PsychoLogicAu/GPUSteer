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



public:
	SteerForSeparationCUDA(	AgentGroup * pAgentGroup, KNNData * pKNNData, AgentGroup * pOtherGroup, float const fWeight );
	virtual ~SteerForSeparationCUDA( void ) {}

	virtual void init( void );
	virtual void run( void );
	virtual void close( void );

};	// class SteerForSeparationCUDA
}	// namespace OpenSteer
#endif