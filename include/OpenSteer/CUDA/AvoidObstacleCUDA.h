#ifndef AVOIDOBSTACLECUDA_H
#define AVOIDOBSTACLECUDA_H

#include "AbstractCUDAKernel.cuh"
#include "../Obstacle.h"

namespace OpenSteer
{
	class AvoidObstacleCUDA : public AbstractCUDAKernel
	{
	protected:
		float						m_minTimeToCollision;
		const SphericalObstacle*	m_pObstacle;

		// Device pointers.
		spherical_obstacle_data*	m_pdObstacleData;

	public:
		AvoidObstacleCUDA(VehicleGroup *pVehicleGroup, const float minTimeToCollision, const SphericalObstacle *pObstacle);
		~AvoidObstacleCUDA(void) {}

		virtual void init(void);
		virtual void run(void);
		virtual void close(void);
	};
} // namespace OpenSteer
#endif