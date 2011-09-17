#ifndef OPENSTEER_CUDASTEERLIBRARY_H
#define OPENSTEER_CUDASTEERLIBRARY_H

#include "../AbstractGroupSteerLibrary.h"

#include "SteerForSeekCUDA.h"
#include "UpdateCUDA.h"
#include "AvoidObstaclesCUDA.h"
#include "AvoidObstacleCUDA.h"
#include "SteerForFleeCUDA.h"
#include "SteerForPursuitCUDA.h"
#include "KNNBruteForceCUDA.cuh"
#include "KNNBinningCUDA.cuh"
#include "SteerToAvoidNeighborsCUDA.cuh"
#include "SteerForSeparationCUDA.cuh"

#include <vector>

#define CUDAGroupSteerLibrary (*CUDAGroupSteerLibrarySingleton::Instance())

namespace OpenSteer
{
class CUDAGroupSteerLibrarySingleton : public AbstractGroupSteerLibrary
{
private:
	static CUDAGroupSteerLibrarySingleton* _instance;
protected:
	CUDAGroupSteerLibrarySingleton(void) {}
public:
	static CUDAGroupSteerLibrarySingleton* Instance(void);

	// AbstractSteerLibrary interface
	// Steering behaviours
	virtual void steerForSeek( AgentGroup * pAgentGroup, const float3 &target, float const fWeight );
	virtual void steerForFlee( AgentGroup * pAgentGroup, const float3 &target, float const fWeight );
	virtual void steerToFollowPath( AgentGroup * pAgentGroup, const float predictionTime, const std::vector<float3> &path, float const fWeight );
	virtual void steerToStayOnPath( AgentGroup * pAgentGroup, const float predictionTime, const std::vector<float3> &path, float const fWeight );

	// Obstacle avoidance
	virtual void steerToAvoidObstacle( AgentGroup * pAgentGroup, const float minTimeToCollision, const SphericalObstacle& obstacle, float const fWeight );
	virtual void steerToAvoidObstacles( AgentGroup * pAgentGroup, const float minTimeToCollision, ObstacleGroup const& obstacles, float const fWeight );

	// Pursuit/ Evasion
	virtual void steerForPursuit( AgentGroup * pAgentGroup, const VehicleData &target, const float maxPredictionTime, float const fWeight );
	virtual void steerForEvasion( AgentGroup * pAgentGroup, const VehicleData &target, const float maxPredictionTime, float const fWeight );

	// KNN search
	virtual void findKNearestNeighbors( AgentGroup * pAgentGroup, KNNData * pKNNData, KNNBinData * pKNNBinData, BaseGroup * pOtherGroup );

	// Update
	virtual void update( AgentGroup * pAgentGroup, const float elapsedTime );

	// Neighborhood based behaviors.
	virtual void steerToAvoidNeighbors( AgentGroup * pAgentGroup, KNNData * pKNNData, AgentGroup * pOtherGroup, const float fMinTimeToCollision, float const fMinSeparationDistance, float const fWeight );

	// Flocking behaviors.
	virtual void steerForSeparation( AgentGroup * pAgentGroup, KNNData * pKNNData, AgentGroup * pOtherGroup, float const fWeight );
	virtual void steerForAlignment(	AgentGroup * pAgentGroup, KNNData * pKNNData, AgentGroup * pOtherGroup, float const fWeight );
	virtual void steerForCohesion( AgentGroup * pAgentGroup, KNNData * pKNNData, AgentGroup * pOtherGroup, float const fWeight );
};	// class CUDAGroupSteerLibrarySingleton
}	// namespace OpenSteer
#endif // GPU_STEER_LIBRARY_H