#ifndef AVOIDOBSTACLESCUDA_H
#define AVOIDOBSTACLESCUDA_H

#include "AbstractCUDAKernel.cuh"
#include "../ObstacleGroup.h"

namespace OpenSteer
{
	typedef struct near_obstacle_index {
		int baseIndex;
		int numObstacles;
	} NearObstacleIndex;

	class AvoidObstaclesCUDA : public AbstractCUDAKernel
	{
	protected:
		float											m_minTimeToCollision;
		ObstacleGroup*									m_pObstacleGroup;
		bool											m_runKernel;

		// Device pointers.
		NearObstacleIndex*								m_pdNearObstacleIndices;
		int*											m_pdObstacleIndices;
		SphericalObstacleData*							m_pdObstacleData;

	public:
		AvoidObstaclesCUDA(VehicleGroup *pVehicleGroup, const float minTimeToCollision, ObstacleGroup *pObstacleGroup);
		~AvoidObstaclesCUDA(void) {}

		virtual void init(void);
		virtual void run(void);
		virtual void close(void);
	};
} // namespace OpenSteer
#endif