#include "CUDAGroupSteerLibrary.h"

using namespace OpenSteer;

CUDAGroupSteerLibrarySingleton* CUDAGroupSteerLibrarySingleton::_instance = NULL;

CUDAGroupSteerLibrarySingleton* CUDAGroupSteerLibrarySingleton::Instance(void)
{
	if(_instance == NULL)
		_instance = new CUDAGroupSteerLibrarySingleton;

	return _instance;
}

void CUDAGroupSteerLibrarySingleton::steerForSeek( AgentGroup * pAgentGroup, float3 const& target, float const fWeight )
{
	SteerForSeekCUDA kernel( pAgentGroup, target, fWeight );

	kernel.init();
	kernel.run();
	kernel.close();
}

// Applies the newly compute steering vector to the vehicles.
void CUDAGroupSteerLibrarySingleton::update( AgentGroup * pAgentGroup, const float elapsedTime )
{
	UpdateCUDA kernel( pAgentGroup, elapsedTime );

	kernel.init();
	kernel.run();
	kernel.close();
}

void CUDAGroupSteerLibrarySingleton::steerForFlee( AgentGroup * pAgentGroup, const float3 &target, float const fWeight )
{
	SteerForFleeCUDA kernel( pAgentGroup, target, fWeight );

	kernel.init();
	kernel.run();
	kernel.close();
}

void CUDAGroupSteerLibrarySingleton::steerToFollowPath(AgentGroup * pAgentGroup, const float predictionTime, const std::vector<float3> &path, float const fWeight)
{

}

void CUDAGroupSteerLibrarySingleton::steerToStayOnPath(AgentGroup * pAgentGroup, const float predictionTime, const std::vector<float3> &path, float const fWeight)
{

}

void CUDAGroupSteerLibrarySingleton::steerToAvoidObstacle( AgentGroup * pAgentGroup, const float minTimeToCollision, const SphericalObstacle& obstacle, float const fWeight )
{
	//AvoidObstacleCUDA kernel( pAgentGroup, minTimeToCollision, &obstacle );

	//kernel.init();
	//kernel.run();
	//kernel.close();
}

void CUDAGroupSteerLibrarySingleton::steerToAvoidObstacles( AgentGroup * pAgentGroup, const float minTimeToCollision, ObstacleGroup const& obstacles, float const fWeight )
{
	//AvoidObstaclesCUDA kernel( pAgentGroup, minTimeToCollision, &obstacles );

	//kernel.init();
	//kernel.run();
	//kernel.close();
}

void CUDAGroupSteerLibrarySingleton::steerToAvoidNeighbors( AgentGroup * pAgentGroup, KNNData * pKNNData, AgentGroup * pOtherGroup, const float fMinTimeToCollision, float const fMinSeparationDistance, float const fWeight )
{
	SteerToAvoidNeighborsCUDA kernel( pAgentGroup, pKNNData, pOtherGroup, fMinTimeToCollision, fMinSeparationDistance, fWeight );

	kernel.init();
	kernel.run();
	kernel.close();
}

void CUDAGroupSteerLibrarySingleton::steerForPursuit( AgentGroup * pAgentGroup, const VehicleData &target, const float maxPredictionTime, float const fWeight )
{
	SteerForPursuitCUDA kernel( pAgentGroup, target.position, target.forward, target.velocity(), target.speed, maxPredictionTime, fWeight );

	kernel.init();
	kernel.run();
	kernel.close();
}

void CUDAGroupSteerLibrarySingleton::findKNearestNeighbors( AgentGroup * pAgentGroup, KNNData * pKNNData, KNNBinData * pKNNBinData, BaseGroup * pOtherGroup )
{
	KNNBinningCUDA kernel( pAgentGroup, pKNNData, pKNNBinData, pOtherGroup );

	//KNNBruteForceCUDA kernel( pAgentGroup, &knnData, &otherGroup );
	//KNNBruteForceCUDAV2 kernel( pAgentGroup, pKNNData, pOtherGroup );
	//KNNBruteForceCUDAV3 kernel( pAgentGroup, &knnData, &otherGroup );

	kernel.init();
	kernel.run();
	kernel.close();
}

void CUDAGroupSteerLibrarySingleton::steerForEvasion( AgentGroup * pAgentGroup, const VehicleData &target, const float maxPredictionTime, float const fWeight )
{

}

void CUDAGroupSteerLibrarySingleton::steerForSeparation(	AgentGroup * pAgentGroup, KNNData * pKNNData, AgentGroup * pOtherGroup, float const fWeight )
{
	SteerForSeparationCUDA kernel( pAgentGroup, pKNNData, pOtherGroup, fWeight );

	kernel.init();
	kernel.run();
	kernel.close();
}

void CUDAGroupSteerLibrarySingleton::steerForAlignment(	AgentGroup * pAgentGroup, KNNData * pKNNData, AgentGroup * pOtherGroup, float const fWeight )
{

}

void CUDAGroupSteerLibrarySingleton::steerForCohesion(	AgentGroup * pAgentGroup, KNNData * pKNNData, AgentGroup * pOtherGroup, float const fWeight )
{

}
