#ifndef OPENSTEER_IGROUPSTEERLIBRARY_H
#define OPENSTEER_IGROUPSTEERLIBRARY_H

#include "Obstacle.h"
#include "ObstacleGroup.h"
#include "AbstractVehicle.h"
#include "VehicleGroup.h"
#include "VectorUtils.cuh"

namespace OpenSteer
{
class AbstractGroupSteerLibrary
{
protected:
public:
	//Steering behaviours
	virtual void steerForSeek(VehicleGroup &vehicleGroup, const float3 &target) = 0;
	virtual void steerForFlee(VehicleGroup &vehicleGroup, const float3 &target) = 0;
	virtual void steerToFollowPath(VehicleGroup &vehicleGroup, const float predictionTime, const std::vector<float3> &path) = 0;
	virtual void steerToStayOnPath(VehicleGroup &vehicleGroup, const float predictionTime, const std::vector<float3> &path) = 0;

	//Obstacle avoidance
	virtual void steerToAvoidObstacle(VehicleGroup &vehicleGroup, const float minTimeToCollision, const SphericalObstacle& obstacle) = 0;
	virtual void steerToAvoidObstacles(VehicleGroup &vehicleGroup, const float minTimeToCollision, ObstacleGroup const& obstacles) = 0;

	//Unaligned collision avoidance
	virtual void steerToAvoidNeighbors(VehicleGroup &vehicleGroup, const float minTimeToCollision, const AVGroup &others) = 0;

	//Pursuit/ Evasion
	virtual void steerForPursuit(VehicleGroup &vehicleGroup, const VehicleData &target, const float maxPredictionTime) = 0;
	virtual void steerForEvasion(VehicleGroup &vehicleGroup, const VehicleData &target, const float maxPredictionTime) = 0;

	// KNN search
	virtual void findKNearestNeighbors( VehicleGroup & vehicleGroup ) = 0;

	// Update
	virtual void update(VehicleGroup &vehicleGroup, const float elapsedTime) = 0;
};	// class AbstractGroupSteerLibrary
}	// namespace OpenSteer
#endif //COMMON_STEERLIBRARY_H