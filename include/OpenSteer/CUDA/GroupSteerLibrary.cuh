#ifndef OPENSTEER_GROUPSTEERLIBRARY_CUH
#define OPENSTEER_GROUPSTEERLIBRARY_CUH

#include "../WallGroup.h"

#include "KNNBruteForceCUDA.cuh"
#include "KNNBinning.cuh"

#include "SteerForSeekCUDA.cuh"
#include "SteerForFleeCUDA.cuh"
#include "SteerForPursueCUDA.cuh"
#include "SteerForEvasionCUDA.cuh"

#include "SteerToAvoidNeighborsCUDA.cuh"
#include "AvoidObstaclesCUDA.cuh"
#include "AvoidWallsCUDA.cuh"

#include "SteerForSeparationCUDA.cuh"
#include "SteerForCohesionCUDA.cuh"
#include "SteerForAlignmentCUDA.cuh"

#include "SteerToFollowPathCUDA.cuh"

#include "AntiPenetrationWallCUDA.cuh"
#include "AntiPenetrationAgentsCUDA.cuh"

#include "UpdateCUDA.h"

#include "WrapWorldCUDA.cuh"

namespace OpenSteer
{

static void wrapWorld( AgentGroup * pAgentGroup, float3 const& worldSize )
{
	WrapWorldCUDA kernel( pAgentGroup, worldSize );

	kernel.init();
	kernel.run();
	kernel.close();
}

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

static void steerForPursuit( AgentGroup * pAgentGroup, float3 const& targetPosition, float3 const& targetDirection, float const& targetSpeed, const float maxPredictionTime, float const fWeight, uint const doNotApplyWith )
{
	SteerForPursueCUDA kernel( pAgentGroup, targetPosition, targetDirection, targetSpeed, maxPredictionTime, fWeight, doNotApplyWith );

	kernel.init();
	kernel.run();
	kernel.close();
}

static void steerForEvade( AgentGroup * pAgentGroup, float3 const& menacePosition, float3 const& menaceDirection, float const menaceSpeed, float const maxPredictionTime, float const fWeight, uint const doNotApplyWith )
{
	SteerForEvadeCUDA kernel( pAgentGroup, menacePosition, menaceDirection, menaceSpeed, maxPredictionTime, fWeight, doNotApplyWith );

	kernel.init();
	kernel.run();
	kernel.close();
}

static void steerToFollowPath(AgentGroup * pAgentGroup, PolylinePathwayCUDA * pPath, float const predictionTime, float const fWeight, uint const doNotApplyWith )
{
	SteerToFollowPathCUDA kernel( pAgentGroup, pPath, predictionTime, fWeight, doNotApplyWith );

	kernel.init();
	kernel.run();
	kernel.close();
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

static void steerToAvoidNeighbors( AgentGroup * pAgentGroup, KNNData * pKNNData, AgentGroup * pOtherGroup, const float fMinTimeToCollision, float const fMinSeparationDistance, bool const bAvoidCloseNeighbors, float const fWeight, uint const doNotApplyWith )
{
	SteerToAvoidNeighborsCUDA kernel( pAgentGroup, pKNNData, pOtherGroup, fMinTimeToCollision, fMinSeparationDistance, bAvoidCloseNeighbors, fWeight, doNotApplyWith );

	kernel.init();
	kernel.run();
	kernel.close();
}

static void steerToAvoidWalls( AgentGroup * pAgentGroup, KNNData * pKNNData, WallGroup * pWallGroup, float const fMinTimeToCollision, float const fWeight, uint const doNotApplyWith )
{
	AvoidWallsCUDA kernel( pAgentGroup, pKNNData, pWallGroup, fMinTimeToCollision, fWeight, doNotApplyWith );

	kernel.init();
	kernel.run();
	kernel.close();
}

static void updateKNNDatabase( BaseGroup * pGroup, KNNBinData * pKNNBinData )
{
	KNNBinningUpdateDBCUDA kernel( pGroup, pKNNBinData );

	kernel.init();
	kernel.run();
	kernel.close();
}

static void findKNearestNeighbors( AgentGroup * pAgentGroup, KNNData * pKNNData, KNNBinData * pKNNBinData, BaseGroup * pOtherGroup, uint const searchRadius )
{
	//KNNBinningCUDA kernel( pAgentGroup, pKNNData, pKNNBinData, pOtherGroup, searchRadius );

	//KNNBruteForceCUDA kernel( pAgentGroup, pKNNData, pOtherGroup );
	//KNNBruteForceCUDAV2 kernel( pAgentGroup, pKNNData, pOtherGroup );
	KNNBruteForceCUDAV3 kernel( pAgentGroup, pKNNData, pOtherGroup );

	kernel.init();
	kernel.run();
	kernel.close();
}



static void steerForSeparation(	AgentGroup * pAgentGroup, KNNData * pKNNData, AgentGroup * pOtherGroup, float const minDistance, float const maxDistance, float const cosMaxAngle, float const fWeight, uint const doNotApplyWith )
{
	SteerForSeparationCUDA kernel( pAgentGroup, pKNNData, pOtherGroup, minDistance, maxDistance, cosMaxAngle, fWeight, doNotApplyWith );

	kernel.init();
	kernel.run();
	kernel.close();
}

static void steerForAlignment( AgentGroup * pAgentGroup, KNNData * pKNNData, AgentGroup * pOtherGroup, float const minDistance, float const maxDistance, float const cosMaxAngle, float const fWeight, uint const doNotApplyWith )
{
	SteerForAlignmentCUDA kernel( pAgentGroup, pKNNData, pOtherGroup, minDistance, maxDistance, cosMaxAngle, fWeight, doNotApplyWith );

	kernel.init();
	kernel.run();
	kernel.close();
}

static void steerForCohesion( AgentGroup * pAgentGroup, KNNData * pKNNData, AgentGroup * pOtherGroup, float const minDistance, float const maxDistance, float const cosMaxAngle, float const fWeight, uint const doNotApplyWith )
{
	SteerForCohesionCUDA kernel( pAgentGroup, pKNNData, pOtherGroup, minDistance, maxDistance, cosMaxAngle, fWeight, doNotApplyWith );

	kernel.init();
	kernel.run();
	kernel.close();
}

// Applies the newly compute steering vector to the vehicles.
static void updateGroup( AgentGroup * pAgentGroup, /*KNNData * pKNNData, WallGroup * pWallGroup,*/ const float elapsedTime )
{
	UpdateCUDA kernel( pAgentGroup, /*pKNNData, pWallGroup,*/ elapsedTime );

	kernel.init();
	kernel.run();
	kernel.close();
}

static void antiPenetrationWall( AgentGroup * pAgentGroup, KNNData * pKNNData, WallGroup * pWallGroup, float const elapsedTime, uint const doNotApplyWith )
{
	AntiPenetrationWALLCUDA kernel( pAgentGroup, pKNNData, pWallGroup, elapsedTime, doNotApplyWith );

	kernel.init();
	kernel.run();
	kernel.close();
}

static void antiPenetrationAgents( AgentGroup * pAgentGroup, KNNData * pKNNData, AgentGroup * pOtherGroup, uint const doNotApplyWith )
{
	AntiPenetrationAgentsCUDA kernel( pAgentGroup, pKNNData, pOtherGroup, doNotApplyWith );

	kernel.init();
	kernel.run();
	kernel.close();
}


}	// namespace OpenSteer

#endif