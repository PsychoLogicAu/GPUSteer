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
		virtual void steerForSeek(VehicleGroup &vehicleGroup, const float3 &target);
		virtual void steerForFlee(VehicleGroup &vehicleGroup, const float3 &target); // TODO: implement
		virtual void steerToFollowPath(VehicleGroup &vehicleGroup, const float predictionTime, const std::vector<float3> &path); // TODO: implement
		virtual void steerToStayOnPath(VehicleGroup &vehicleGroup, const float predictionTime, const std::vector<float3> &path); // TODO: implement

		//Obstacle avoidance
		virtual void steerToAvoidObstacle(VehicleGroup &vehicleGroup, const float minTimeToCollision, const SphericalObstacle& obstacle); // TODO: implement
		virtual void steerToAvoidObstacles(VehicleGroup &vehicleGroup, const float minTimeToCollision, const ObstacleGroup &obstacles); // TODO: implement

		//Unaligned collision avoidance
		virtual void steerToAvoidNeighbors(VehicleGroup &vehicleGroup, const float minTimeToCollision, const AVGroup &others); // TODO: implement

		//Pursuit/ Evasion
		virtual void steerForPursuit(VehicleGroup &vehicleGroup, const VehicleData &target, const float maxPredictionTime); // TODO: implement
		virtual void steerForEvasion(VehicleGroup &vehicleGroup, const VehicleData &target, const float maxPredictionTime); // TODO: implement

		// Update
		virtual void update(VehicleGroup &vehicleGroup, const float elapsedTime);
	};
}
#endif //CPU_STEER_LIBRARY_H