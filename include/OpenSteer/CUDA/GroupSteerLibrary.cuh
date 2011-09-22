#ifndef OPENSTEER_GROUPSTEERLIBRARY_CUH
#define OPENSTEER_GROUPSTEERLIBRARY_CUH

#include "SteerForSeekCUDA.h"
#include "UpdateCUDA.h"
#include "AvoidObstaclesCUDA.cuh"
#include "SteerForFleeCUDA.h"
#include "SteerForPursueCUDA.h"
#include "KNNBruteForceCUDA.cuh"
#include "KNNBinningCUDA.cuh"
#include "SteerToAvoidNeighborsCUDA.cuh"
#include "SteerForSeparationCUDA.cuh"

namespace OpenSteer
{

static void steerForSeek( AgentGroup * pAgentGroup, float3 const& target, float const fWeight, uint const doNotApplyWith )
{
	SteerForSeekCUDA kernel( pAgentGroup, target, fWeight, doNotApplyWith );

	kernel.init();
	kernel.run();
	kernel.close();
}

static void steerForFlee( AgentGroup * pAgentGroup, const float3 &target, float const fWeight, uint const doNotApplyWith )
{
	SteerForFleeCUDA kernel( pAgentGroup, target, fWeight, doNotApplyWith );

	kernel.init();
	kernel.run();
	kernel.close();
}

static void steerForPursuit( AgentGroup * pAgentGroup, const VehicleData &target, const float maxPredictionTime, float const fWeight, uint const doNotApplyWith )
{
	SteerForPursueCUDA kernel( pAgentGroup, target.position, target.forward, target.velocity(), target.speed, maxPredictionTime, fWeight, doNotApplyWith );

	kernel.init();
	kernel.run();
	kernel.close();
}

static void steerForEvade( AgentGroup * pAgentGroup, const VehicleData &target, const float maxPredictionTime, float const fWeight )
{

}

static void steerToFollowPath(AgentGroup * pAgentGroup, const float predictionTime, const std::vector<float3> &path, float const fWeight)
{

}

static void steerToStayOnPath(AgentGroup * pAgentGroup, const float predictionTime, const std::vector<float3> &path, float const fWeight)
{

}

static void steerToAvoidObstacles( AgentGroup * pAgentGroup, ObstacleGroup * pObstacleGroup, KNNData * pKNNData, float const fMinTimeToCollision, float const fWeight, uint const doNotApplyWith )
{
	AvoidObstaclesCUDA kernel( pAgentGroup, pObstacleGroup, pKNNData, fMinTimeToCollision, fWeight, doNotApplyWith );

	kernel.init();
	kernel.run();
	kernel.close();
}

static void steerToAvoidNeighbors( AgentGroup * pAgentGroup, KNNData * pKNNData, AgentGroup * pOtherGroup, const float fMinTimeToCollision, float const fMinSeparationDistance, float const fWeight, uint const doNotApplyWith )
{
	SteerToAvoidNeighborsCUDA kernel( pAgentGroup, pKNNData, pOtherGroup, fMinTimeToCollision, fMinSeparationDistance, fWeight, doNotApplyWith );

	kernel.init();
	kernel.run();
	kernel.close();
}

static void findKNearestNeighbors( AgentGroup * pAgentGroup, KNNData * pKNNData, KNNBinData * pKNNBinData, BaseGroup * pOtherGroup )
{
	KNNBinningCUDA kernel( pAgentGroup, pKNNData, pKNNBinData, pOtherGroup );

	//KNNBruteForceCUDA kernel( pAgentGroup, &knnData, &otherGroup );
	//KNNBruteForceCUDAV2 kernel( pAgentGroup, pKNNData, pOtherGroup );
	//KNNBruteForceCUDAV3 kernel( pAgentGroup, &knnData, &otherGroup );

	kernel.init();
	kernel.run();
	kernel.close();
}



static void steerForSeparation(	AgentGroup * pAgentGroup, KNNData * pKNNData, AgentGroup * pOtherGroup, float const fWeight, uint const doNotApplyWith )
{
	SteerForSeparationCUDA kernel( pAgentGroup, pKNNData, pOtherGroup, fWeight, doNotApplyWith );

	kernel.init();
	kernel.run();
	kernel.close();
}

static void steerForAlignment(	AgentGroup * pAgentGroup, KNNData * pKNNData, AgentGroup * pOtherGroup, float const fWeight, uint const doNotApplyWith )
{

}

static void steerForCohesion(	AgentGroup * pAgentGroup, KNNData * pKNNData, AgentGroup * pOtherGroup, float const fWeight, uint const doNotApplyWith )
{

}

// Applies the newly compute steering vector to the vehicles.
static void updateGroup( AgentGroup * pAgentGroup, const float elapsedTime )
{
	UpdateCUDA kernel( pAgentGroup, elapsedTime );

	kernel.init();
	kernel.run();
	kernel.close();
}


}	// namespace OpenSteer

#endif