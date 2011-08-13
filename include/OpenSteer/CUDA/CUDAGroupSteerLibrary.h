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
	virtual void steerForSeek( VehicleGroup &vehicleGroup, const float3 &target );
	virtual void steerForFlee( VehicleGroup &vehicleGroup, const float3 &target );
	virtual void steerToFollowPath( VehicleGroup &vehicleGroup, const float predictionTime, const std::vector<float3> &path );
	virtual void steerToStayOnPath( VehicleGroup &vehicleGroup, const float predictionTime, const std::vector<float3> &path );

	// Obstacle avoidance
	virtual void steerToAvoidObstacle( VehicleGroup &vehicleGroup, const float minTimeToCollision, const SphericalObstacle& obstacle );
	virtual void steerToAvoidObstacles( VehicleGroup &vehicleGroup, const float minTimeToCollision, ObstacleGroup const& obstacles );

	// Unaligned collision avoidance
	virtual void steerToAvoidNeighbors( VehicleGroup &vehicleGroup, const float minTimeToCollision, const AVGroup &others );

	// Pursuit/ Evasion
	virtual void steerForPursuit( VehicleGroup &vehicleGroup, const VehicleData &target, const float maxPredictionTime );
	virtual void steerForEvasion( VehicleGroup &vehicleGroup, const VehicleData &target, const float maxPredictionTime );

	// KNN search
	virtual void findKNearestNeighbors( VehicleGroup & vehicleGroup, size_t const k );

	// Update
	virtual void update( VehicleGroup &vehicleGroup, const float elapsedTime );
};	// class CUDAGroupSteerLibrarySingleton
}	// namespace OpenSteer
#endif // GPU_STEER_LIBRARY_H