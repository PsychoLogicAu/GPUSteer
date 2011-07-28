#ifndef UPDATECUDA_H
#define UPDATECUDA_H

#include "AbstractCUDAKernel.h"

namespace OpenSteer
{
	class UpdateCUDA : public AbstractCUDAKernel
	{
	protected:
		float m_elapsedTime;

	public:
		UpdateCUDA(VehicleGroup *pVehicleGroup, const float elapsedTime);
		~UpdateCUDA(void) {}

		virtual void init(void);
		virtual void run(void);
		virtual void close(void);
	};
}

#endif