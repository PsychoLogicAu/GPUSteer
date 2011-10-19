#ifndef OPENSTEER_SIMULATION_H
#define OPENSTEER_SIMULATION_H

#include "OpenSteer/CUDA/CUDAGlobals.cuh"

#include <string>
#include <fstream>
#include <vector>

namespace OpenSteer
{

class GroupParams
{
protected:
	uint					m_nNumAgents;
	std::vector< float3 >	m_vecPathPoints;
	bool					m_bPathIsCyclic;

public:
	GroupParams( void )
	:	m_nNumAgents( 0 ),
		m_vecPathPoints(),
		m_bPathIsCyclic( false )
	{}
	virtual ~GroupParams( void )
	{}

	void open( std::ifstream & fs );
};	// class BaseGroup

class WorldParams
{
protected:
	float3				m_f3Dimensions;
	uint3				m_u3Cells;
	uint				m_nObstacleCount;
	std::string			m_strMapFilename;

public:
	WorldParams( void )
	{}
	virtual ~WorldParams( void )
	{}

	void open( std::ifstream & fs );
};	// class BaseWorld

class Simulation
{
protected:
	WorldParams 					m_WorldParams;
	std::vector< GroupParams * >	m_vecGroups;

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



public:
	Simulation( void );
	virtual ~Simulation( void );

	virtual void open( char const* szFilename );
	virtual void reset( void );
	virtual void close( void );
	virtual void redraw( const float currentTime, const float elapsedTime );
	virtual void update( const float currentTime, const float elapsedTime );


};	// class Simulation




}	// namespace OpenSteer
#endif
