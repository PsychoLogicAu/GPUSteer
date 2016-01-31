#ifndef OPENSTEER_SIMULATION_H
#define OPENSTEER_SIMULATION_H

#include "OpenSteer/Globals.h"
#include "OpenSteer/CUDA/CUDAGlobals.cuh"

#include "OpenSteer/AgentGroup.h"

#include <string>
#include <fstream>
#include <vector>

namespace OpenSteer
{

class GroupParams
{
public:
	GroupParams( void )
	:	m_nNumAgents( 0 ),
		m_vecPathPoints(),
		m_bPathIsCyclic( false )
	{}
	virtual ~GroupParams( void )
	{}

	void load( std::ifstream & fs );

	// Number of agents.
	uint					m_nNumAgents;

	float					m_fMaxSpeed;
	float					m_fMaxForce;

	// Starting position of group.
	float3					m_f3StartPosition;
	float					m_fMinStartRadius;
	float					m_fMaxStartRadius;

	// Path.
	std::vector< float3 >	m_vecPathPoints;
	float					m_fPathRadius;
	bool					m_bPathIsCyclic;

	float3					m_f3BodyColor;
};	// class BaseGroup

class WorldParams
{
public:
	WorldParams( void )
	{}
	virtual ~WorldParams( void )
	{}

	void load( std::ifstream & fs );

	float3				m_f3Dimensions;
	uint3				m_u3Cells;
	uint				m_nObstacleCount;
	std::string			m_strMapFilename;
};	// class BaseWorld

class SimulationParams
{
protected:
public:
	SimulationParams( void ){}
	~SimulationParams( void ) {}

	void load( char const* szFilename );

	WorldParams 				m_WorldParams;
	std::vector< GroupParams >	m_vecGroupParams;

	uint	m_nSeed;
	uint	m_nKNN;
	uint	m_nKNO;
	uint	m_nKNW;
	uint	m_nSearchRadius;
	float	m_fMaxPursuitPredictionTime;
	float	m_fMinTimeToCollision;
	float	m_fMinTimeToObstacle;
	float	m_fMinTimeToWall;
	float	m_fPathPredictionTime;
	float	m_fMinSeparationDistance;
	float	m_fMaxSeparationDistance;
	float	m_fMinFlockingDistance;
	float	m_fMaxFlockingDistance;
	float	m_fCosMaxFlockingAngle;
	bool	m_bAvoidCloseNeighbors;
	//	Behavior	weights
	float	m_fWeightAlignment;
	float	m_fWeightCohesion;
	float	m_fWeightSeparation;
	float	m_fWeightPursuit;
	float	m_fWeightSeek;
	float	m_fWeightFlee;
	float	m_fWeightEvade;
	float	m_fWeightFollowPath;
	float	m_fWeightAvoidObstacles;
	float	m_fWeightAvoidWalls;
	float	m_fWeightAvoidNeighbors;
	//	Masks
	uint	m_nMaskAlignment;
	uint	m_nMaskAntiPenetrationAgents;
	uint	m_nMaskCohesion;
	uint	m_nMaskSeparation;
	uint	m_nMaskPursuit;
	uint	m_nMaskSeek;
	uint	m_nMaskFlee;
	uint	m_nMaskEvade;
	uint	m_nMaskFollowPath;
	uint	m_nMaskAvoidObstacles;
	uint	m_nMaskAvoidWalls;
	uint	m_nMaskAvoidNeighbors;
};	// class SimulationParams

class Simulation
{
protected:
	SimulationParams	m_SimulationParams;

public:
	Simulation( void )
	:	m_SimulationParams()
	{}

	virtual ~Simulation( void )
	{}

	virtual void load( char const* szFilename )
	{
		m_SimulationParams.load( szFilename );
	}

	virtual void reset( void )
	{
		m_SimulationParams.m_vecGroupParams.clear();
	}

	SimulationParams const& Params( void ) { return m_SimulationParams; }
};

class SimulationWorld
{
	friend class SimulationGroup;
protected:
	WorldParams *		m_pWorldParams;
	SimulationParams *	m_pSimulationParams;

public:
	SimulationWorld( SimulationParams * pSimulationParams, WorldParams * pWorldParams )
	:	m_pWorldParams( pWorldParams ),
		m_pSimulationParams( pSimulationParams )
	{}

	virtual ~SimulationWorld( void )
	{}
};

class SimulationGroup : public AgentGroup
{
	friend class SimulationWorld;
protected:
	GroupParams *		m_pGroupParams;
	SimulationParams *	m_pSimulationParams;

public:
	SimulationGroup( SimulationParams * pSimulationParams, GroupParams * pGroupParams )
	:	AgentGroup( pSimulationParams->m_WorldParams.m_u3Cells, pSimulationParams->m_nKNN ),
		m_pGroupParams( pGroupParams ),
		m_pSimulationParams( pSimulationParams )
	{ }

	virtual ~SimulationGroup( void )
	{ }

	virtual void reset(void) = 0;
	virtual void draw(void) = 0;
	virtual void update(const float currentTime, const float elapsedTime) = 0;
};

}	// namespace OpenSteer
#endif
