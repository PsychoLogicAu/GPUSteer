// ----------------------------------------------------------------------------
//
//
// OpenSteer -- Steering Behaviors for Autonomous Characters
//
// Copyright (c) 2002-2003, Sony Computer Entertainment America
// Original author: Craig Reynolds <craig_reynolds@playstation.sony.com>
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
// THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS IN THE SOFTWARE.
//
//
// ----------------------------------------------------------------------------
//
//
// Capture the Flag   (a portion of the traditional game)
//
// The "Capture the Flag" sample steering problem, proposed by Marcin
// Chady of the Working Group on Steering of the IGDA's AI Interface
// Standards Committee (http://www.igda.org/Committees/ai.htm) in this
// message (http://sourceforge.net/forum/message.php?msg_id=1642243):
//
//     "An agent is trying to reach a physical location while trying
//     to stay clear of a group of enemies who are actively seeking
//     him. The environment is littered with obstacles, so collision
//     avoidance is also necessary."
//
// Note that the enemies do not make use of their knowledge of the 
// seeker's goal by "guarding" it.  
//
// XXX hmm, rename them "attacker" and "defender"?
//
// 08-12-02 cwr: created 
//
//
// ----------------------------------------------------------------------------


#include <iomanip>
#include <sstream>
#include <fstream>
#include <list>

#include "OpenSteer/Globals.h"

#include "OpenSteer/Simulation.h"

#include "OpenSteer/SimpleVehicle.h"
#include "OpenSteer/OpenSteerDemo.h"
#include "OpenSteer/Proximity.h"
#include "OpenSteer/ObstacleGroup.h"

#include "OpenSteer/AgentGroup.h"

#include "OpenSteer/CUDA/CUDAKernelGlobals.cuh"
#include "OpenSteer/CUDA/GroupSteerLibrary.cuh"

#include "OpenSteer/AgentData.h"

// Include the required KNN headers.
#include "OpenSteer/CUDA/KNNBinData.cuh"

#include "OpenSteer/WallGroup.h"

#include "OpenSteer/CUDA/PolylinePathwayCUDA.cuh"

using namespace OpenSteer;

#ifndef min
#define min(a,b)            (((a) < (b)) ? (a) : (b))
#endif

#define M_PI       3.14159265358979323846

//#define ANNOTATION_LINES
//#define ANNOTATION_TEXT
//#define ANNOTATION_CELLS
//#define NO_DRAW
//#define ONLY_DRAW_NEAR_CAMERA

// ----------------------------------------------------------------------------
// forward declarations
class BoidsGroup;
class BoidsBase;
//class BoidsObstacleGroup;
class BoidsProxy;
class BoidsWorld;
class BoidsSimulation;

// ----------------------------------------------------------------------------
// state for OpenSteerDemo PlugIn
//
// XXX consider moving this inside CtfPlugIn
// XXX consider using STL (any advantage? consistency?)
//BoidsGroup *			g_pBoids;
//BoidsObstacleGroup *	g_pObstacles;
//BoidsWorld *			g_pWorld;
//BoidsProxy *			g_pCameraProxy;

BoidsSimulation *		g_pSimulation;

// ----------------------------------------------------------------------------
// globals

#pragma region obsolete
// Using cell diameter of 7
/*
const int	gEnemyCount					= 100;
const float	gDim						= 63;
const int	gCells						= 10;

//const int gEnemyCount					= 1000;
//const float gDim						= 200;
//const int gCells						= 28;

//const int gEnemyCount					= 10000;
//const float gDim						= 635;
//const int gCells						= 150;


//const int gEnemyCount					= 100000;
//const float gDim						= 2000;
////const int gCells						= 285;
//const int gCells						= 170;

const int gEnemyCount					= 100000;
const float gDim						= 2000;
//const int gCells						= 285;
const int gCells						= 400;


const int gEnemyCount					= 1000000;
const float gDim						= 6350;
////const int gCells						= 907;
//const int gCells						= 1814;
const int gCells						= 400;


//const int gEnemyCount					= 1000;
//const float gDim						= 100;
//const int gCells						= 25;

const float3 gWorldSize					= make_float3( gDim, gDim, gDim );
const uint3 gWorldCells					= make_uint3( gCells, gCells, gCells );

const int gObstacleCount				= 1;

uint const	g_knn						= 5;		// Number of near neighbors to keep track of.
uint const	g_kno						= 2;		// Number of near obstacles to keep track of.
uint const	g_knw						= 3;		// Number of near walls to keep track of.
uint const	g_maxSearchRadius			= 1;		// Distance in cells for the maximum search radius.
uint const	g_searchRadiusNeighbors		= 1;		// Distance in cells to search for neighbors.
uint const	g_searchRadiusObstacles		= 1;		// Distance in cells to search for obstacles.

float const g_fMaxPursuitPredictionTime	= 10.0f;		// Look-ahead time for pursuit.
float const g_fMinSeparationDistance	= 0.5f;		// Agents will steer hard to avoid other agents within this radius, and brake if other agent is ahead.
float const g_fMinTimeToCollision		= 2.0f;		// Look-ahead time for neighbor avoidance.
float const g_fMinTimeToObstacle		= 5.0f;		// Look-ahead time for obstacle avoidance.
float const g_fMinTimeToWall			= 5.0f;		// Look-ahead time for wall avoidance.
float const g_fPathPredictionTime		= 10.f;

// Weights for behaviors.
float const	g_fWeightAlignment			= 16.f;
float const	g_fWeightCohesion			= 16.f;
float const	g_fWeightSeparation			= 16.f;

float const g_fWeightPursuit			= 1.f;
float const g_fWeightSeek				= 10.f;
float const g_fWeightEvade				= 10.f;

float const g_fWeightFollowPath			= 6.f;

float const g_fWeightObstacleAvoidance	= 10.f;
float const g_fWeightWallAvoidance		= 10.f;
float const g_fWeightAvoidNeighbors		= 2.f;

// Masks for behaviors.
uint const	g_maskAlignment				= KERNEL_SEPARATION_BIT;//KERNEL_SEPARATION_BIT;
uint const	g_maskCohesion				= KERNEL_SEPARATION_BIT;//KERNEL_SEPARATION_BIT;
uint const	g_maskSeparation			= 0;

uint const	g_maskSeek					= KERNEL_SEPARATION_BIT;
uint const	g_maskFlee					= KERNEL_AVOID_OBSTACLES_BIT | KERNEL_AVOID_WALLS_BIT | KERNEL_AVOID_NEIGHBORS_BIT;
uint const	g_maskPursuit				= KERNEL_SEPARATION_BIT;//KERNEL_AVOID_OBSTACLES_BIT | KERNEL_AVOID_WALLS_BIT | KERNEL_AVOID_NEIGHBORS_BIT;
uint const	g_maskEvade					= KERNEL_AVOID_OBSTACLES_BIT | KERNEL_AVOID_WALLS_BIT | KERNEL_AVOID_NEIGHBORS_BIT;

uint const	g_maskFollowPath			= KERNEL_AVOID_OBSTACLES_BIT;

uint const	g_maskObstacleAvoidance		= 0;
uint const	g_maskNeighborAvoidance		= KERNEL_AVOID_OBSTACLES_BIT;
uint const	g_maskWallAvoidance			= 0;


//float const g_fMaxFlockingDistance		= 2 * g_searchRadiusNeighbors * gDim / gCells;
//float const g_fCosMaxFlockingAngle		= 360 * (float)M_PI / 180.f;	// 350 degrees - used in "An efficient GPU implementation for large scale individual-based simulation of collective behavior"
float const g_fMinFlockingDistance		= 0.5f;
float const g_fMaxSeparationDistance	= 1.f;
//float const g_fMaxFlockingDistance		= 7.f;
float const g_fMaxFlockingDistance		= FLT_MAX;

//float const g_fCosMaxFlockingAngle		= cosf( 2 * (float)M_PI );
float const g_fCosMaxFlockingAngle		= 0.98480775301220805936674302458952f;


// Start position.
float3 const g_f3StartBaseCenter		= make_float3( 0.f, 0.f, 0.f );
float const g_fMinStartRadius			= 0.0f;
float const g_fMaxStartRadius			= 0.5f * min( gWorldSize.x, min( gWorldSize.y, gWorldSize.z ) );
*/
#pragma endregion

const float gBrakingRate				= 0.75f;

const float3 evadeColor					= make_float3(0.6f, 0.6f, 0.3f); // annotation
const float3 seekColor					= make_float3(0.3f, 0.6f, 0.6f); // annotation
const float3 clearPathColor				= make_float3(0.3f, 0.6f, 0.3f); // annotation

const float gAvoidancePredictTimeMin	= 0.9f;
const float gAvoidancePredictTimeMax	= 2.0f;
float const gAvoidancePredictTime		= gAvoidancePredictTimeMin;

// Function prototypes.
void randomizeStartingPositionAndHeadingBoids( float4 & position, float3 & up, float4 & forward, float3 & side, float const minRadius, float const maxRadius, float3 const& startPosition );

//void randomizeStartingPositionAndHeading(VehicleData &vehicleData, VehicleConst &vehicleConst)
void randomizeStartingPositionAndHeadingBoids( float4 & position, float3 & up, float4 & forward, float3 & side, float const minRadius, float const maxRadius, float3 const& startPosition )
{
    // randomize position on a ring between inner and outer radii
    // centered around the home base
    const float rRadius = frandom2 ( minRadius, maxRadius );
	float3 const randomOnSphere = float3_scalar_multiply( float3_RandomVectorInUnitRadiusSphere(), rRadius );

    position = make_float4( float3_add( startPosition, randomOnSphere ), 0.f );

	float3 newForward;
    randomizeHeading( up, newForward, side );
	forward = make_float4( newForward, 0.f );
}

class BoidsWorld : public SimulationWorld
{
private:
	// Bin data to be used for KNN lookups.
	KNNBinData *							m_pKNNBinData;

public:
	BoidsWorld( SimulationParams * pSimulationParams, WorldParams * pWorldParams )
	:	SimulationWorld( pSimulationParams, pWorldParams ),
		m_pKNNBinData( NULL )
	{
		m_pKNNBinData = new KNNBinData( m_pWorldParams->m_u3Cells, m_pWorldParams->m_f3Dimensions, pSimulationParams->m_nSearchRadius );
	}

	~BoidsWorld( void )
	{
		SAFE_DELETE( m_pKNNBinData ); 
	}

	void draw( void )
	{
#if defined ANNOTATION_CELLS
		//
		//	Draw the cells.
		//
		std::vector< bin_cell > const& cells	= m_pKNNBinData->hvCells();
		float3 const cellColor = { 0.1f, 0.1f, 0.1f };
		// For each of the cells...
		for( std::vector< bin_cell >::const_iterator it = cells.begin(); it != cells.end(); ++it )
		{
			
			//  4     5
			//  +----+
			//0/|  1/|
			//+-+--+ |
			//| +--+-+
			//|/6  |/7
			//+----+
			//2    3
			
			float3 const p0		= { it->minBound.x, it->maxBound.y, it->minBound.z };
			float3 const p1		= { it->maxBound.x, it->maxBound.y, it->minBound.z };
			float3 const p2		= { it->minBound.x, it->minBound.y, it->minBound.z };
			float3 const p3		= { it->maxBound.x, it->minBound.y, it->minBound.z };
			float3 const p4		= { it->minBound.x, it->maxBound.y, it->maxBound.z };
			float3 const p5		= { it->maxBound.x, it->maxBound.y, it->maxBound.z };
			float3 const p6		= { it->minBound.x, it->minBound.y, it->maxBound.z };
			float3 const p7		= { it->maxBound.x, it->minBound.y, it->maxBound.z };

			drawLine( p0, p1, cellColor );
			drawLine( p0, p2, cellColor );
			drawLine( p0, p4, cellColor );
			drawLine( p1, p3, cellColor );
			drawLine( p1, p5, cellColor );
			drawLine( p2, p3, cellColor );
			drawLine( p2, p6, cellColor );
			drawLine( p3, p7, cellColor );
			drawLine( p4, p5, cellColor );
			drawLine( p4, p6, cellColor );
			drawLine( p5, p7, cellColor );
			drawLine( p6, p7, cellColor );
		}
#endif
	}

	KNNBinData * GetBinData( void )		{ return m_pKNNBinData; }
};




// ----------------------------------------------------------------------------
// This PlugIn uses two vehicle types: CtfSeeker and CtfEnemy.  They have a
// common base class: BoidsBase which is a specialization of SimpleVehicle.
class BoidsBase : public SimpleVehicle
{
public:
    // constructor
    BoidsBase () {reset ();}

    // reset state
    void reset (void);

    // draw this character/vehicle into the scene
    void draw (void);

    // for draw method
    float3 bodyColor;
};

class BoidsGroup : public SimulationGroup
{
private:
	KNNData *				m_pKNNSelf;
	KNNData *				m_pKNNObstacles;

public:
	BoidsGroup( SimulationParams * pSimulationParams, GroupParams * pGroupParams )
	:	SimulationGroup( pSimulationParams, pGroupParams ),
		m_pKNNSelf( NULL ),
		m_pKNNObstacles( NULL )
	{
		m_pKNNSelf = new KNNData( m_pGroupParams->m_nNumAgents, m_pSimulationParams->m_nKNN );
		m_pKNNObstacles = new KNNData( m_pGroupParams->m_nNumAgents, m_pSimulationParams->m_nKNO );

		reset();
	}
	virtual ~BoidsGroup(void)
	{
		SAFE_DELETE( m_pKNNSelf );
		SAFE_DELETE( m_pKNNObstacles );
	}

	void reset(void);
	void draw(void);

	void update(const float currentTime, const float elapsedTime);
};

class BoidsProxy : public BoidsBase
{
public:
    // constructor
    BoidsProxy (){}

	// Pull the latest data from the selected agent.
    void update( uint const selectedAgent, BoidsGroup * pBoids )
	{
		pBoids->GetAgentGroupData().getAgentData( selectedAgent, _data );
	}
};

class BoidsSimulation : public Simulation
{
	friend class BoidsGroup;
	friend class BoidsPlugIn;
protected:
	BoidsWorld *	m_pWorld;
	BoidsProxy *	m_pCameraProxy;
	BoidsGroup *	m_pBoids;



public:
	BoidsSimulation( void )
	:	m_pCameraProxy( NULL ),
		m_pWorld( NULL ),
		m_pBoids( NULL )
	{
		m_pCameraProxy = new BoidsProxy;
	}
	virtual ~BoidsSimulation( void )
	{
		SAFE_DELETE( m_pCameraProxy );
		SAFE_DELETE( m_pWorld );
		SAFE_DELETE( m_pBoids );
	}

	virtual void load( char const* szFilename )
	{
		Simulation::load( szFilename );

		srand( m_SimulationParams.m_nSeed );
		OpenSteerDemo::maxSelectedVehicleIndex = m_SimulationParams.m_vecGroupParams[0].m_nNumAgents;

		SAFE_DELETE( m_pWorld );
		SAFE_DELETE( m_pBoids );

		m_pWorld = new BoidsWorld( &m_SimulationParams, &m_SimulationParams.m_WorldParams );
		m_pBoids = new BoidsGroup( &m_SimulationParams, &m_SimulationParams.m_vecGroupParams[0] );
	}

	void update( float const currentTime, float const elapsedTime )
	{
		// Update the boids group
		m_pBoids->update(currentTime, elapsedTime);

		// Update the camera proxy object.
		m_pCameraProxy->update( OpenSteerDemo::selectedVehicleIndex, m_pBoids );
	}

	void draw( void )
	{
        // draw the enemy
		m_pBoids->draw();

		// draw the world
		m_pWorld->draw();

		// display status in the upper left corner of the window
		std::ostringstream status;
		//status << std::left << std::setw( 25 ) << "No. obstacles: " << std::setw( 10 ) << g_pObstacles->Size() << std::endl;
		status << std::left << std::setw( 25 ) << "No. agents: " << m_pBoids->Size() << std::endl;
		status << std::left << std::setw( 25 ) << "World dim: " << m_SimulationParams.m_WorldParams.m_f3Dimensions << std::endl;
		status << std::left << std::setw( 25 ) << "World cells: " << m_SimulationParams.m_WorldParams.m_u3Cells << std::endl;
		status << std::left << std::setw( 25 ) << "Search radius: " << m_SimulationParams.m_nSearchRadius << std::endl;
		status << std::left << std::setw( 25 ) << "Camera proxy position: " << make_float3( m_pCameraProxy->position() ) << std::endl;
		const float h = drawGetWindowHeight ();
		const float3 screenLocation = make_float3(10, h-50, 0);
		draw2dTextAt2dLocation (status, screenLocation, gGray80);
	}
};

/*
#define testOneObstacleOverlap(radius, center)					\
{																\
    float d = float3_distance (od.position, center);			\
    float clearance = d - (od.radius + (radius));				\
    if (minClearance > clearance) minClearance = clearance;		\
}

class BoidsObstacleGroup : public ObstacleGroup
{
private:
	void addOneObstacle (void)
	{
		std::vector< float4 > & positions = m_obstacleGroupData.hvPosition();
		std::vector< float > & radii = m_obstacleGroupData.hvRadius();

		float minClearance;
		const float requiredClearance = 2.0f; //gSeeker->radius() * 4; // 2 x diameter

		ObstacleData od;

		do
		{
			minClearance = FLT_MAX;

			od.radius = frandom2 (1.5f, 4.0f); // random radius between 1.5 and 4
			//od.position = float3_scalar_multiply(float3_randomVectorOnUnitRadiusXZDisk(), g_fMaxStartRadius * 1.1f);
			od.position = make_float4( float3_scalar_multiply(float3_RandomVectorInUnitRadiusSphere(), g_fMaxStartRadius * 0.9f), 0.f );

			// Make sure it doesn't overlap with any of the other obstacles.
			for( size_t i = 0; i < Size(); i++ )
			{
				float d = float3_distance( make_float3(od.position), make_float3(positions[i]) );
				float clearance = d - (od.radius + radii[i]);
				if ( clearance < minClearance )
					minClearance = clearance;
			}
		} while (minClearance < requiredClearance);

		// add new non-overlapping obstacle to registry
		AddObstacle( od );
	}

public:
	BoidsObstacleGroup( uint3 const& worldCells, uint const kno )
	:	ObstacleGroup( worldCells, kno )
	{
	}
	virtual ~BoidsObstacleGroup( void ) {}

	void reset(void)
	{
		while( Size() < gObstacleCount )
			addOneObstacle();

		// Send the data to the device.
		SyncDevice();

		// Update the KNN database for the group.
		updateKNNDatabase( this, g_pWorld->GetBinData() );
	}

	void draw(void)
	{
        const float3 color = make_float3(0.8f, 0.6f, 0.4f);

		for( uint i = 0; i < m_nCount; i++ )
		{
			ObstacleData od;
			GetDataForObstacle( i, od );

			//drawXZCircle( od.radius, od.position, color, 20 );
			draw3dCircle( od.radius, make_float3(od.position), make_float3(OpenSteerDemo::camera.forward()), color, 20 );
		}
	}
};
*/

void BoidsGroup::reset(void)
{
	Clear();
	//static unsigned int id = 0;
	// Add the required number of enemies.
	while( Size() < m_pGroupParams->m_nNumAgents )
	{
		BoidsBase boid;
		AgentData &aData = boid.getVehicleData();

		aData.speed = 3.0f;
		aData.maxForce = 3.0f;
		aData.maxSpeed = 3.0f;
		randomizeStartingPositionAndHeadingBoids( aData.position, aData.up, aData.direction, aData.side, m_pGroupParams->m_fMinStartRadius, m_pGroupParams->m_fMaxStartRadius, m_pGroupParams->m_f3StartPosition );
		
		bool success = AddAgent( aData );
	}

	// Transfer the data to the device.
	SyncDevice();

	// Compute the initial KNN for this group with itself.
	// Update the KNN database for the group.
	updateKNNDatabase( this, g_pSimulation->m_pWorld->GetBinData() );
	findKNearestNeighbors( this, m_pKNNSelf, g_pSimulation->m_pWorld->GetBinData(), this, m_pSimulationParams->m_nSearchRadius );
}

void BoidsGroup::draw(void)
{
#if defined NO_DRAW
	return;
#endif

#if defined ONLY_DRAW_NEAR_CAMERA
	float3 const cameraPosition = make_float3( OpenSteerDemo::camera.position() );
#endif
	// Draw all of the enemies
	float3 bodyColor = make_float3(0.6f, 0.4f, 0.4f); // redish

	AgentData ad;

	AgentGroupData & m_agentGroupData = GetAgentGroupData();

#if defined ANNOTATION_LINES || defined ANNOTATION_TEXT
	// Temporary storage used for annotation.
	uint KNNIndices[g_knn];
	float KNNDistances[g_knn];
#endif

	// For each enemy...
	for( size_t i = 0; i < Size(); i++ )
	{
		// Get its varialbe and constant data.
		m_agentGroupData.getAgentData( i, ad );

#if defined ONLY_DRAW_NEAR_CAMERA
		if( float3_distanceSquared( make_float3( ad.position ), cameraPosition ) > 10000.f )
			continue;

#endif

		// Draw the agent.
		drawBasic3dSphericalVehicle( ad.radius, make_float3(ad.position), make_float3(ad.direction), ad.side, ad.up, bodyColor );

#if defined ANNOTATION_LINES || defined ANNOTATION_TEXT
		//
		// Annotation
		//

		// Pull the KNN data for this agent from the nearest neighbor database.
		m_pKNNSelf->getAgentData( i, KNNIndices, KNNDistances );
		
#endif

#if defined ANNOTATION_TEXT
		// annotate the agent with useful data.
		const float3 textOrigin = float3_add( make_float3(ad.position), make_float3( 0, 0.25, 0 ) );
		std::ostringstream annote;

		// Write this agent's index.
		annote << i << std::endl;
		// Write each ID of the KNN for this agent.
		//if( i % 5 == 0 )
		//{
		//	for( uint j = 0; j < g_knn; j++ )
		//		annote << KNNIndices[j] << " ";
		//}
		annote << std::ends;

		draw2dTextAt3dLocation (annote, textOrigin, gWhite);
#endif

#if defined ANNOTATION_LINES
		AgentData adOther;
		float3 const lineColor = make_float3( 1.f, 1.f, 1.f );

		// Draw the KNN links.
		for( uint j = 0; j < g_knn; j++ )
		{
			if( KNNIndices[j] < g_pBoids->Size() )
			{
				m_agentGroupData.getAgentData( KNNIndices[j], adOther );
				drawLine( make_float3(ad.position), make_float3(adOther.position), lineColor );
			}
		}
#endif
	}
}

void BoidsGroup::update(const float currentTime, const float elapsedTime)
{
	wrapWorld( this, m_pSimulationParams->m_WorldParams.m_f3Dimensions );

	// Update the positions in the KNNDatabase for the group.
	updateKNNDatabase( this, g_pSimulation->m_pWorld->GetBinData() );

	// Update the KNNDatabases
	//findKNearestNeighbors( this, m_pKNNObstacles, g_pSimulation->m_pWorld->GetBinData(), gObstacles, g_searchRadiusObstacles );
	findKNearestNeighbors( this, m_pKNNSelf, g_pSimulation->m_pWorld->GetBinData(), this, m_pSimulationParams->m_nSearchRadius );

	// Avoid collision with obstacles.
	//steerToAvoidObstacles( this, gObstacles, m_pKNNObstacles, g_fMinTimeToObstacle, g_fWeightObstacleAvoidance, g_maskObstacleAvoidance );

	// Avoid collision with self.
	//steerToAvoidNeighbors( this, m_pKNNSelf, this,  g_fMinTimeToCollision, g_fMinSeparationDistance, g_fWeightAvoidNeighbors, g_maskNeighborAvoidance );

	// Flocking.
	steerForSeparation( this, m_pKNNSelf, this, m_pSimulationParams->m_fMinFlockingDistance, m_pSimulationParams->m_fMaxSeparationDistance, m_pSimulationParams->m_fCosMaxFlockingAngle, m_pSimulationParams->m_fWeightSeparation, m_pSimulationParams->m_nMaskSeparation );
	steerForCohesion( this, m_pKNNSelf, this, m_pSimulationParams->m_fMinFlockingDistance, m_pSimulationParams->m_fMaxFlockingDistance, m_pSimulationParams->m_fCosMaxFlockingAngle, m_pSimulationParams->m_fWeightCohesion, m_pSimulationParams->m_nMaskCohesion );
	steerForAlignment( this, m_pKNNSelf, this, m_pSimulationParams->m_fMinFlockingDistance, m_pSimulationParams->m_fMaxFlockingDistance, m_pSimulationParams->m_fCosMaxFlockingAngle, m_pSimulationParams->m_fWeightAlignment, m_pSimulationParams->m_nMaskAlignment );

	// Pursue target.
	//steerForPursuit( this, gSeeker->getVehicleData(), g_fMaxPursuitPredictionTime, g_fWeightPursuit, g_maskPursuit );
	//steerForPursuit( this, g_pWanderer->getVehicleData(), g_fMaxPursuitPredictionTime, g_fWeightPursuit, g_maskPursuit );
	//steerForSeek( this, make_float3( g_pWanderer->position() ), g_fWeightSeek, g_maskSeek );

	//steerForEvade( this, g_pWanderer->position(), g_pWanderer->forward(), g_pWanderer->speed(), g_fMaxPursuitPredictionTime, g_fWeightEvade, g_maskEvade );

	// Apply steering.
	updateGroup( this, elapsedTime );
}

// ----------------------------------------------------------------------------
// reset state
void BoidsBase::reset (void)
{
	//_data.id = serialNumber;
	_data.id = serialNumber;

    SimpleVehicle::reset ();  // reset the vehicle 

    setSpeed (3);             // speed along Forward direction.
    setMaxForce (3.0);        // steering force is clipped to this magnitude
    setMaxSpeed (3.0);        // velocity is clipped to this magnitude

    clearTrailHistory ();     // prevent long streaks due to teleportation
}

// ----------------------------------------------------------------------------
// draw this character/vehicle into the scene
void BoidsBase::draw (void)
{
    drawBasic2dCircularVehicle (*this, bodyColor);
    //drawTrail ();
}


// ----------------------------------------------------------------------------
// PlugIn for OpenSteerDemo


class BoidsPlugIn : public PlugIn
{
private:

public:

    const char* name (void) {return "Boids";}

    float selectionOrderSortKey (void) {return 0.01f;}

    virtual ~BoidsPlugIn() {} // be more "nice" to avoid a compiler warning

    void open (void)
    {
		OpenSteerDemo::setAnnotationOff();

		int numDevices;
		cudaGetDeviceCount(&numDevices);

		// TODO: more intelligent selection of the CUDA device.
		CUDA_SAFE_CALL( cudaSetDevice( 0 ) );

		g_pSimulation = new BoidsSimulation;

		// initialize camera
		OpenSteerDemo::init3dCamera ( *g_pSimulation->m_pCameraProxy );
		OpenSteerDemo::camera.mode = Camera::cmFixedDistanceOffset;
        OpenSteerDemo::camera.fixedDistDistance = 100.f;
        OpenSteerDemo::camera.fixedDistVOffset = 1.f;
        OpenSteerDemo::camera.lookdownDistance = 20;
        OpenSteerDemo::camera.aimLeadTime = 0.5;
        OpenSteerDemo::camera.povOffset = make_float3( 0, 0.5, -2 );

		reset();

		all.push_back( g_pSimulation->m_pCameraProxy );
    }

	void reset (void)
    {
		g_pSimulation->reset();
		g_pSimulation->load( "BoidsGroup.params" );



        // reset camera position
		//OpenSteerDemo::camera.reset();

        // make camera jump immediately to new position
        OpenSteerDemo::camera.doNotSmoothNextMove ();
    }

    void update (const float currentTime, const float elapsedTime)
    {
		g_pSimulation->update( currentTime, elapsedTime );
	}

    void redraw (const float currentTime, const float elapsedTime)
    {
		// update camera
		OpenSteerDemo::updateCamera (currentTime, elapsedTime, *g_pSimulation->m_pCameraProxy);

		g_pSimulation->draw();
    }

    void close (void)
    {
		SAFE_DELETE( g_pSimulation );
    }

    void handleFunctionKeys (int keyNumber)
    {
        //switch (keyNumber)
        //{
		//case 1: addOneObstacle();
		//	break;
		//case 2: removeOneObstacle();
		//	break;
        //}
    }

    void printMiniHelpForFunctionKeys (void)
    {
        std::ostringstream message;
        message << "Function keys handled by ";
        message << '"' << name() << '"' << ':' << std::ends;
        OpenSteerDemo::printMessage (message);
        //OpenSteerDemo::printMessage ("  F1     add one obstacle.");
        //OpenSteerDemo::printMessage ("  F2     remove one obstacle.");
        OpenSteerDemo::printMessage ("");
    }

	const AVGroup& allVehicles (void) {return (const AVGroup&) all;}

	// a group (STL vector) of all vehicles in the PlugIn
    std::vector<BoidsBase*> all;
};


BoidsPlugIn gCtfPlugIn;


// ----------------------------------------------------------------------------


