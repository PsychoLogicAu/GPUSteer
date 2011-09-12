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
	virtual void steerForSeek( VehicleGroup &vehicleGroup, const float3 &target, float const fWeight );
	virtual void steerForFlee( VehicleGroup &vehicleGroup, const float3 &target, float const fWeight );
	virtual void steerToFollowPath( VehicleGroup &vehicleGroup, const float predictionTime, const std::vector<float3> &path, float const fWeight );
	virtual void steerToStayOnPath( VehicleGroup &vehicleGroup, const float predictionTime, const std::vector<float3> &path, float const fWeight );

	// Obstacle avoidance
	virtual void steerToAvoidObstacle( VehicleGroup &vehicleGroup, const float minTimeToCollision, const SphericalObstacle& obstacle, float const fWeight );
	virtual void steerToAvoidObstacles( VehicleGroup &vehicleGroup, const float minTimeToCollision, ObstacleGroup const& obstacles, float const fWeight );

	// Pursuit/ Evasion
	virtual void steerForPursuit( VehicleGroup &vehicleGroup, const VehicleData &target, const float maxPredictionTime, float const fWeight );
	virtual void steerForEvasion( VehicleGroup &vehicleGroup, const VehicleData &target, const float maxPredictionTime, float const fWeight );

	// KNN search
	virtual void findKNearestNeighbors( VehicleGroup & vehicleGroup );

	// Update
	virtual void update( VehicleGroup &vehicleGroup, const float elapsedTime );

	// Neighborhood based behaviors.
	virtual void steerToAvoidNeighbors(VehicleGroup &vehicleGroup, const float fMinTimeToCollision, float const fMinSeparationDistance, float const fWeight );

	// Flocking behaviors.
	virtual void steerForSeparation( VehicleGroup &vehicleGroup, float const fWeight );
	virtual void steerForAlignment( VehicleGroup &vehicleGroup, float const fWeight );
	virtual void steerForCohesion( VehicleGroup &vehicleGroup, float const fWeight );
};	// class CUDAGroupSteerLibrarySingleton
}	// namespace OpenSteer
#endif // GPU_STEER_LIBRARY_H