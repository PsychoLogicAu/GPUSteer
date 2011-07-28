#ifndef STEERFORPURSUITCUDA_H
#define STEERFORPURSUITCUDA_H

#include "AbstractCUDAKernel.h"

//#include "../VehicleData.cu"

namespace OpenSteer
{
	class SteerForPursuitCUDA : public AbstractCUDAKernel
	{
	protected:
		float					m_maxPredictionTime;
		const vehicle_data*		m_pTarget;

		// Device pointers.
		vehicle_data*			m_pdTarget;

	public:
		SteerForPursuitCUDA(VehicleGroup *pVehicleGroup, const vehicle_data *pTarget, const float maxPredictionTime);
		~SteerForPursuitCUDA(void) {}

		virtual void init(void);
		virtual void run(void);
		virtual void close(void);
	};
} // namespace OpenSteer
#endif