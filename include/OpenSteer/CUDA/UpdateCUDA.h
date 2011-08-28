#ifndef UPDATECUDA_H
#define UPDATECUDA_H

#include "AbstractCUDAKernel.cuh"

namespace OpenSteer
{
	class UpdateCUDA : public AbstractCUDAKernel
	{
	protected:
		float m_fElapsedTime;

	public:
		UpdateCUDA( VehicleGroup * pVehicleGroup, const float fElapsedTime );
		~UpdateCUDA( void ) {}

		virtual void init( void );
		virtual void run( void );
		virtual void close( void );
	};
}

#endif