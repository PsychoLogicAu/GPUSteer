#ifndef STEERFORFLEECUDA_H
#define STEERFORFLEECUDA_H

#include "AbstractCUDAKernel.cuh"

namespace OpenSteer
{
	class SteerForFleeCUDA : public AbstractCUDAKernel
	{
	protected:
		float3		m_target;

	public:
		SteerForFleeCUDA( VehicleGroup * pVehicleGroup, const float3 &target );
		~SteerForFleeCUDA( void ) {}

		virtual void init( void );
		virtual void run( void );
		virtual void close( void );
	};
} // namespace OpenSteer
#endif