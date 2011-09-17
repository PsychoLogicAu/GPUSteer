#include "CUDAGroupSteerLibrary.h"

using namespace OpenSteer;

CUDAGroupSteerLibrarySingleton* CUDAGroupSteerLibrarySingleton::_instance = NULL;

CUDAGroupSteerLibrarySingleton* CUDAGroupSteerLibrarySingleton::Instance(void)
{
	if(_instance == NULL)
		_instance = new CUDAGroupSteerLibrarySingleton;

	return _instance;
}

void CUDAGroupSteerLibrarySingleton::steerForSeek( AgentGroup & agentGroup, float3 const& target, float const fWeight )
{
	//agentGroup.OutputDataToFile("vehicledata.txt");
	SteerForSeekCUDA kernel( &agentGroup, target, fWeight );

	kernel.init();
	kernel.run();
	kernel.close();
}

// Applies the newly compute steering vector to the vehicles.
void CUDAGroupSteerLibrarySingleton::update( AgentGroup &agentGroup, const float elapsedTime )
{
	UpdateCUDA kernel( &agentGroup, elapsedTime );

	kernel.init();
	kernel.run();
	kernel.close();
}

void CUDAGroupSteerLibrarySingleton::steerForFlee( AgentGroup &agentGroup, const float3 &target, float const fWeight )
{
	SteerForFleeCUDA kernel( &agentGroup, target, fWeight );

	kernel.init();
	kernel.run();
	kernel.close();
}

void CUDAGroupSteerLibrarySingleton::steerToFollowPath(AgentGroup &agentGroup, const float predictionTime, const std::vector<float3> &path, float const fWeight)
{

}

void CUDAGroupSteerLibrarySingleton::steerToStayOnPath(AgentGroup &agentGroup, const float predictionTime, const std::vector<float3> &path, float const fWeight)
{

}

void CUDAGroupSteerLibrarySingleton::steerToAvoidObstacle( AgentGroup &agentGroup, const float minTimeToCollision, const SphericalObstacle& obstacle, float const fWeight )
{
	//AvoidObstacleCUDA kernel( &agentGroup, minTimeToCollision, &obstacle );

	//kernel.init();
	//kernel.run();
	//kernel.close();
}

void CUDAGroupSteerLibrarySingleton::steerToAvoidObstacles( AgentGroup &agentGroup, const float minTimeToCollision, ObstacleGroup const& obstacles, float const fWeight )
{
	//AvoidObstaclesCUDA kernel( &agentGroup, minTimeToCollision, &obstacles );

	//kernel.init();
	//kernel.run();
	//kernel.close();
}

void CUDAGroupSteerLibrarySingleton::steerToAvoidNeighbors( AgentGroup &agentGroup, const float fMinTimeToCollision, float const fMinSeparationDistance, float const fWeight )
{
	SteerToAvoidNeighborsCUDA kernel( &agentGroup, fMinTimeToCollision, fMinSeparationDistance, fWeight );

	kernel.init();
	kernel.run();
	kernel.close();
}

void CUDAGroupSteerLibrarySingleton::steerForPursuit( AgentGroup &agentGroup, const VehicleData &target, const float maxPredictionTime, float const fWeight )
{
	SteerForPursuitCUDA kernel( &agentGroup, target.position, target.forward, target.velocity(), target.speed, maxPredictionTime, fWeight );

	kernel.init();
	kernel.run();
	kernel.close();
}

void CUDAGroupSteerLibrarySingleton::findKNearestNeighbors( AgentGroup & agentGroup, KNNData & knnData, KNNBinData & knnBinData, BaseGroup & otherGroup )
{
	KNNBinningCUDA kernel( &agentGroup, &knnData, &knnBinData, &otherGroup );

	//KNNBruteForceCUDA kernel( &agentGroup, &knnData, &otherGroup );
	//KNNBruteForceCUDAV2 kernel( &agentGroup, &knnData, &otherGroup );
	//KNNBruteForceCUDAV3 kernel( &agentGroup, &knnData, &otherGroup );

	kernel.init();
	kernel.run();
	kernel.close();
}

void CUDAGroupSteerLibrarySingleton::steerForEvasion( AgentGroup &agentGroup, const VehicleData &target, const float maxPredictionTime, float const fWeight )
{

}

void CUDAGroupSteerLibrarySingleton::steerForSeparation( AgentGroup &agentGroup, float const fWeight )
{
	SteerForSeparationCUDA kernel( &agentGroup, fWeight );

	kernel.init();
	kernel.run();
	kernel.close();
}

void CUDAGroupSteerLibrarySingleton::steerForAlignment( AgentGroup &agentGroup, float const fWeight )
{

}

void CUDAGroupSteerLibrarySingleton::steerForCohesion( AgentGroup &agentGroup, float const fWeight )
{

}
