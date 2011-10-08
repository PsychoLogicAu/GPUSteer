#ifndef STEERFORPURSUITCUDA_H
#define STEERFORPURSUITCUDA_H

#include "AbstractCUDAKernel.cuh"

namespace OpenSteer
{
class SteerForPursueCUDA : public AbstractCUDAKernel
{
protected:
	float					m_fMaxPredictionTime;

	float3					m_targetPosition;
	float3					m_targetDirection;
	float3					m_targetVelocity;
	float					m_targetSpeed;

public:
	SteerForPursueCUDA(		AgentGroup * pAgentGroup, 
							float3 const& targetPosition, float3 const& targetDirection, float const& targetSpeed,
							const float fMaxPredictionTime,
							float const fWeight, uint const doNotApplyWith
							);
	~SteerForPursueCUDA( void ) {}

	virtual void init( void );
	virtual void run( void );
	virtual void close( void );
};	// class SteerForPursueCUDA
} // namespace OpenSteer
#endif
