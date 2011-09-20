#ifndef AVOIDOBSTACLESCUDA_H
#define AVOIDOBSTACLESCUDA_H

#include "AbstractCUDAKernel.cuh"

#include "../ObstacleGroup.h"

#include "KNNData.cuh"

namespace OpenSteer
{
	class AvoidObstaclesCUDA : public AbstractCUDAKernel
	{
	protected:
		float				m_fMinTimeToCollision;
		ObstacleGroup *		m_pObstacleGroup;

		KNNData *			m_pKNNData;

	public:
		AvoidObstaclesCUDA( AgentGroup * pAgentGroup, ObstacleGroup * pObstacleGroup, KNNData * pKNNData, float const fMinTimeToCollision, float const fWeight );
		virtual ~AvoidObstaclesCUDA(void) {}

		virtual void init(void);
		virtual void run(void);
		virtual void close(void);
	};
} // namespace OpenSteer
#endif