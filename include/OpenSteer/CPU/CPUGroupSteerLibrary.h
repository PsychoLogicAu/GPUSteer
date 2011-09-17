#ifndef OPENSTEER_CPUSTEERLIBRARY_H
#define OPENSTEER_CPUSTEERLIBRARY_H

#include "../AbstractGroupSteerLibrary.h"
#include <vector>

namespace OpenSteer
{
class CPUGroupSteerLibrary : public AbstractGroupSteerLibrary
{
protected:
public:
	//AbstractSteerLibrary interface
	//Steering behaviours
	virtual void steerForSeek(AgentGroup &agentGroup, const float3 &target);
	virtual void steerForFlee(AgentGroup &agentGroup, const float3 &target); // TODO: implement
	virtual void steerToFollowPath(AgentGroup &agentGroup, const float predictionTime, const std::vector<float3> &path); // TODO: implement
	virtual void steerToStayOnPath(AgentGroup &agentGroup, const float predictionTime, const std::vector<float3> &path); // TODO: implement

	//Obstacle avoidance
	virtual void steerToAvoidObstacle(AgentGroup &agentGroup, const float minTimeToCollision, const SphericalObstacle& obstacle); // TODO: implement
	virtual void steerToAvoidObstacles(AgentGroup &agentGroup, const float minTimeToCollision, ObstacleGroup const& obstacles); // TODO: implement

	//Unaligned collision avoidance
	virtual void steerToAvoidNeighbors( AgentGroup &agentGroup, const float fMinTimeToCollision, float const fMinSeparationDistance );

	//Pursuit/ Evasion
	virtual void steerForPursuit(AgentGroup &agentGroup, const VehicleData &target, const float maxPredictionTime); // TODO: implement
	virtual void steerForEvasion(AgentGroup &agentGroup, const VehicleData &target, const float maxPredictionTime); // TODO: implement

	// KNN search
	virtual void findKNearestNeighbors( AgentGroup & agentGroup );

	// Update
	virtual void update(AgentGroup &agentGroup, const float elapsedTime);
};	// class CPUGroupSteerLibrary
};	// namespace OpenSteer
#endif //CPU_STEER_LIBRARY_H