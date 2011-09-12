#ifndef STEERFORPURSUITCUDA_H
#define STEERFORPURSUITCUDA_H

#include "AbstractCUDAKernel.cuh"

namespace OpenSteer
{
class SteerForPursuitCUDA : public AbstractCUDAKernel
{
protected:
	float					m_fMaxPredictionTime;

	float3					m_targetPosition;
	float3					m_targetForward;
	float3					m_targetVelocity;
	float					m_targetSpeed;

public:
	SteerForPursuitCUDA(	VehicleGroup * pVehicleGroup, 
							float3 const& targetPosition, float3 const& targetForward, float3 const& targetVelocity, float const& targetSpeed,
							const float fMaxPredictionTime, float const fWeight );
	~SteerForPursuitCUDA( void ) {}

	virtual void init( void );
	virtual void run( void );
	virtual void close( void );
};	// class SteerForPursuitCUDA
} // namespace OpenSteer
#endif
