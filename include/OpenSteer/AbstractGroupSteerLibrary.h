#ifndef OPENSTEER_IGROUPSTEERLIBRARY_H
#define OPENSTEER_IGROUPSTEERLIBRARY_H

#include "Obstacle.h"
#include "ObstacleGroup.h"
#include "AbstractVehicle.h"
#include "AgentGroup.h"
#include "VectorUtils.cuh"

#include "KNNData.cuh"
#include "KNNBinData.cuh"

namespace OpenSteer
{
class AbstractGroupSteerLibrary
{
protected:
public:
	//Steering behaviours
	virtual void steerForSeek(AgentGroup &agentGroup, const float3 &target, float const fWeight) = 0;
	virtual void steerForFlee(AgentGroup &agentGroup, const float3 &target, float const fWeight) = 0;
	virtual void steerToFollowPath(AgentGroup &agentGroup, const float predictionTime, const std::vector<float3> &path, float const fWeight) = 0;
	virtual void steerToStayOnPath(AgentGroup &agentGroup, const float predictionTime, const std::vector<float3> &path, float const fWeight) = 0;

	//Obstacle avoidance
	virtual void steerToAvoidObstacle(AgentGroup &agentGroup, const float minTimeToCollision, const SphericalObstacle& obstacle, float const fWeight) = 0;
	virtual void steerToAvoidObstacles(AgentGroup &agentGroup, const float minTimeToCollision, ObstacleGroup const& obstacles, float const fWeight) = 0;

	//Pursuit/ Evasion
	virtual void steerForPursuit(AgentGroup &agentGroup, const VehicleData &target, const float maxPredictionTime, float const fWeight) = 0;
	virtual void steerForEvasion(AgentGroup &agentGroup, const VehicleData &target, const float maxPredictionTime, float const fWeight) = 0;

	// KNN search
	virtual void findKNearestNeighbors( AgentGroup & agentGroup, KNNData & knnData, KNNBinData & knnBinData, BaseGroup & otherGroup ) = 0;

	// Update
	virtual void update(AgentGroup &agentGroup, const float elapsedTime) = 0;

	// Neighborhood based behaviors.
	virtual void steerToAvoidNeighbors( AgentGroup &agentGroup, const float fMinTimeToCollision, float const fMinSeparationDistance, float const fWeight ) = 0;

	// Flocking behaviors.
	virtual void steerForSeparation( AgentGroup &agentGroup, float const fWeight ) = 0;
	virtual void steerForAlignment( AgentGroup &agentGroup, float const fWeight ) = 0;
	virtual void steerForCohesion( AgentGroup &agentGroup, float const fWeight ) = 0;

	// 
};	// class AbstractGroupSteerLibrary
}	// namespace OpenSteer
#endif //COMMON_STEERLIBRARY_H