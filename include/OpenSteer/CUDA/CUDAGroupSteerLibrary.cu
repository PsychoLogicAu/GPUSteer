#include "CUDAGroupSteerLibrary.h"

using namespace OpenSteer;

CUDAGroupSteerLibrarySingleton* CUDAGroupSteerLibrarySingleton::_instance = NULL;

CUDAGroupSteerLibrarySingleton* CUDAGroupSteerLibrarySingleton::Instance(void)
{
	if(_instance == NULL)
		_instance = new CUDAGroupSteerLibrarySingleton;

	return _instance;
}

void CUDAGroupSteerLibrarySingleton::steerForSeek(VehicleGroup &vehicleGroup, const float3 &target)
{
	//vehicleGroup.OutputDataToFile("vehicledata.txt");
	SteerForSeekCUDA kernel( &vehicleGroup, target );

	kernel.init();
	kernel.run();
	kernel.close();
}

// Applies the newly compute steering vector to the vehicles.
void CUDAGroupSteerLibrarySingleton::update( VehicleGroup &vehicleGroup, const float elapsedTime )
{
	UpdateCUDA kernel( &vehicleGroup, elapsedTime );

	kernel.init();
	kernel.run();
	kernel.close();
}

void CUDAGroupSteerLibrarySingleton::steerForFlee( VehicleGroup &vehicleGroup, const float3 &target )
{
	SteerForFleeCUDA kernel( &vehicleGroup, target );

	kernel.init();
	kernel.run();
	kernel.close();
}

void CUDAGroupSteerLibrarySingleton::steerToFollowPath(VehicleGroup &vehicleGroup, const float predictionTime, const std::vector<float3> &path)
{

}

void CUDAGroupSteerLibrarySingleton::steerToStayOnPath(VehicleGroup &vehicleGroup, const float predictionTime, const std::vector<float3> &path)
{

}

void CUDAGroupSteerLibrarySingleton::steerToAvoidObstacle( VehicleGroup &vehicleGroup, const float minTimeToCollision, const SphericalObstacle& obstacle )
{
	//AvoidObstacleCUDA kernel( &vehicleGroup, minTimeToCollision, &obstacle );

	//kernel.init();
	//kernel.run();
	//kernel.close();
}

void CUDAGroupSteerLibrarySingleton::steerToAvoidObstacles( VehicleGroup &vehicleGroup, const float minTimeToCollision, ObstacleGroup const& obstacles )
{
	//AvoidObstaclesCUDA kernel( &vehicleGroup, minTimeToCollision, &obstacles );

	//kernel.init();
	//kernel.run();
	//kernel.close();
}

void CUDAGroupSteerLibrarySingleton::steerToAvoidNeighbors( VehicleGroup &vehicleGroup, const float minTimeToCollision, const AVGroup &others )
{
}

void CUDAGroupSteerLibrarySingleton::steerForPursuit( VehicleGroup &vehicleGroup, const VehicleData &target, const float maxPredictionTime )
{
	SteerForPursuitCUDA kernel( &vehicleGroup, target.position, target.forward, target.velocity(), target.speed, maxPredictionTime );

	kernel.init();
	kernel.run();
	kernel.close();
}

void CUDAGroupSteerLibrarySingleton::findKNearestNeighbors( VehicleGroup & vehicleGroup )
{
	
	//KNNBinningCUDA kernel( &vehicleGroup, k );

	//KNNBruteForceCUDA kernel( &vehicleGroup, k );
	//KNNBruteForceCUDAV2 kernel( &vehicleGroup, k );


	KNNBruteForceCUDAV3 kernel( &vehicleGroup );

	kernel.init();
	kernel.run();
	kernel.close();
}

void CUDAGroupSteerLibrarySingleton::steerForEvasion( VehicleGroup &vehicleGroup, const VehicleData &target, const float maxPredictionTime )
{

}