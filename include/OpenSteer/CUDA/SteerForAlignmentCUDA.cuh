#ifndef OPENSTEER_STEERFORALIGNMENTCUDE_CUH
#define OPENSTEER_STEERFORALIGNMENTCUDE_CUH

#include "AbstractCUDAKernel.cuh"

#include "KNNData.cuh"

namespace OpenSteer
{
class SteerForAlignmentCUDA : public AbstractCUDAKernel
{
protected:
	KNNData *		m_pKNNData;
	AgentGroup *	m_pOtherGroup;

	float			m_fMaxDistance;
	float			m_fCosMaxAngle;

public:
	SteerForAlignmentCUDA(	AgentGroup * pAgentGroup, KNNData * pKNNData, AgentGroup * pOtherGroup, float const maxDistance, float const cosMaxAngle, float const fWeight, uint const doNotApplyWith );
	virtual ~SteerForAlignmentCUDA( void ) {}

	virtual void init( void );
	virtual void run( void );
	virtual void close( void );

};	// class SteerForAlignmentCUDA
}	// namespace OpenSteer
#endif
