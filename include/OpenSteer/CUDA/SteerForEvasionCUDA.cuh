#ifndef OPENSTEER_STEERFOREVADECUDA_CUH
#define OPENSTEER_STEERFOREVADECUDA_CUH

#include "AbstractCUDAKernel.cuh"

namespace OpenSteer
{
class SteerForEvadeCUDA : public AbstractCUDAKernel
{
protected:
	float3		m_menacePosition;
	float3		m_menaceDirection;
	float		m_menaceSpeed;

	float		m_fMaxPredictionTime;

public:
	SteerForEvadeCUDA( AgentGroup * pAgentGroup, float3 const& menacePosition, float3 const& menaceDirection, float const menaceSpeed, float const fMaxPredictionTime, float const fWeight, uint const doNotApplyWith );
	~SteerForEvadeCUDA( void ) {}

	virtual void init( void );
	virtual void run( void );
	virtual void close( void );
};	// class SteerForEvadeCUDA
}	// namespace OpenSteer
#endif
