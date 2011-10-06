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

// ----------------------------------------------------------------------------
// forward declarations
class BoidsGroup;
class BoidsBase;
class BoidsObstacleGroup;
class BoidsWanderer;

#define SAFE_DELETE( x )	{ if( x ){ delete x; x = NULL; } }

// ----------------------------------------------------------------------------
// globals


// Using cell diameter of 7

//const int	gEnemyCount					= 100;
//const float	gDim						= 63;
//const int	gCells						= 10;

//const int gEnemyCount					= 1000;
//const float gDim						= 200;
//const int gCells						= 28;

//const int gEnemyCount					= 10000;
//const float gDim						= 635;
//const int gCells						= 91;


const int gEnemyCount					= 100000;
const float gDim						= 2000;
//const int gCells						= 285;
const int gCells						= 150;


//const int gEnemyCount					= 1000000;
//const float gDim						= 6350;
////const int gCells						= 907;
//const int gCells						= 1814;



//const int gEnemyCount					= 1000;
//const float gDim						= 100;
//const int gCells						= 25;

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
float const	g_fWeightSeparation			= 1.f;

float const g_fWeightPursuit			= 1.f;
float const g_fWeightSeek				= 10.f;
float const g_fWeightEvade				= 10.f;

float const g_fWeightFollowPath			= 6.f;

float const g_fWeightObstacleAvoidance	= 10.f;
float const g_fWeightWallAvoidance		= 10.f;
float const g_fWeightAvoidNeighbors		= 2.f;

// Masks for behaviors.
uint const	g_maskAlignment				= KERNEL_SEPARATION_BIT;
uint const	g_maskCohesion				= KERNEL_SEPARATION_BIT;
uint const	g_maskSeparation			= 0;

uint const	g_maskSeek					= KERNEL_SEPARATION_BIT; //KERNEL_AVOID_OBSTACLES_BIT | KERNEL_AVOID_WALLS_BIT /*| KERNEL_AVOID_NEIGHBORS_BIT*/;
uint const	g_maskFlee					= KERNEL_AVOID_OBSTACLES_BIT | KERNEL_AVOID_WALLS_BIT | KERNEL_AVOID_NEIGHBORS_BIT;
uint const	g_maskPursuit				= KERNEL_SEPARATION_BIT;//KERNEL_AVOID_OBSTACLES_BIT | KERNEL_AVOID_WALLS_BIT | KERNEL_AVOID_NEIGHBORS_BIT;
uint const	g_maskEvade					= KERNEL_AVOID_OBSTACLES_BIT | KERNEL_AVOID_WALLS_BIT | KERNEL_AVOID_NEIGHBORS_BIT;

uint const	g_maskFollowPath			= /*KERNEL_AVOID_WALLS_BIT |*/ KERNEL_AVOID_OBSTACLES_BIT;

uint const	g_maskObstacleAvoidance		= 0;
uint const	g_maskNeighborAvoidance		= KERNEL_AVOID_OBSTACLES_BIT /*| KERNEL_AVOID_WALLS_BIT*/;
uint const	g_maskWallAvoidance			= 0;


//float const g_fMaxFlockingDistance		= 2 * g_searchRadiusNeighbors * gDim / gCells;
//float const g_fCosMaxFlockingAngle		= 360 * (float)M_PI / 180.f;	// 350 degrees - used in "An efficient GPU implementation for large scale individual-based simulation of collective behavior"
float const g_fMinFlockingDistance		= 0.5f;
float const g_fMaxSeparationDistance	= 2.f;
float const g_fMaxFlockingDistance		= 7.f;

//float const g_fCosMaxFlockingAngle		= cosf( 2 * (float)M_PI );
float const g_fCosMaxFlockingAngle		= 0.98480775301220805936674302458952f;

const float3 gWorldSize					= make_float3( gDim, gDim, gDim );
const uint3 gWorldCells					= make_uint3( gCells, gCells, gCells );

// Start position.
float3 const g_f3StartBaseCenter		= make_float3( 0.f, 0.f, 0.f );
float const g_fMinStartRadius			= 0.0f;
float const g_fMaxStartRadius			= 0.15f * min( gWorldSize.x, min( gWorldSize.y, gWorldSize.z ) );

const float gBrakingRate				= 0.75f;

const float3 evadeColor					= make_float3(0.6f, 0.6f, 0.3f); // annotation
const float3 seekColor					= make_float3(0.3f, 0.6f, 0.6f); // annotation
const float3 clearPathColor				= make_float3(0.3f, 0.6f, 0.3f); // annotation

const float gAvoidancePredictTimeMin	= 0.9f;
const float gAvoidancePredictTimeMax	= 2.0f;
float const gAvoidancePredictTime		= gAvoidancePredictTimeMin;

// Function prototypes.
void randomizeStartingPositionAndHeadingBoids( float3 & position, float const radius, float3 & up, float3 & forward, float3 & side );

//void randomizeStartingPositionAndHeading(VehicleData &vehicleData, VehicleConst &vehicleConst)
void randomizeStartingPositionAndHeadingBoids( float3 & position, float const radius, float3 & up, float3 & forward, float3 & side )
{
    // randomize position on a ring between inner and outer radii
    // centered around the home base
    const float rRadius = frandom2 ( g_fMinStartRadius, g_fMaxStartRadius );
	float3 const randomOnSphere = float3_scalar_multiply( float3_RandomVectorInUnitRadiusSphere(), rRadius );

    position =  float3_add( g_f3StartBaseCenter, randomOnSphere );

    randomizeHeading( up, forward, side );
}

class BoidsWorld
{
private:
// Bin data to be used for KNN lookups.
	KNNBinData *							m_pKNNBinData;

public:
	BoidsWorld( uint3 const& worldCells, float3 const& worldSize, uint const maxSearchRadius )
	:	m_pKNNBinData( NULL )
	{
		m_pKNNBinData = new KNNBinData( worldCells, worldSize, maxSearchRadius );
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
// state for OpenSteerDemo PlugIn
//
// XXX consider moving this inside CtfPlugIn
// XXX consider using STL (any advantage? consistency?)
BoidsGroup *			g_pBoids;
//BoidsObstacleGroup *	g_pObstacles;
BoidsWorld *			g_pWorld;
BoidsWanderer *			g_pWanderer;

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

class BoidsWanderer : public BoidsBase
{
public:
    // constructor
    BoidsWanderer () {reset ();}

    // reset state
    void reset (void)
	{
		randomizeStartingPositionAndHeadingBoids( position(), radius(), up(), forward(), side() );
		setPosition( make_float3( 0.f, 0.f, 0.f ) );
	}

    // per frame simulation update
    void update (const float currentTime, const float elapsedTime )
	{
		float3 steer;
		
		float const halfDim = 0.4f * gDim;
		float3 const& position = _data.position;

		if( position.x < -halfDim || position.x > halfDim ||
			position.y < -halfDim || position.y > halfDim ||
			position.z < -halfDim || position.z > halfDim  )
		{
			// Outside of the world bounds. Seek back in.
			steer = steerForSeek( *this, make_float3( 0.f, 0.f, 0.f ) );
		}
		else
		{
			// Inside of the world bonds. Wander.
			steer = steerForWander( *this, elapsedTime );
		}

		applySteeringForce (steer, elapsedTime);
	}

    void draw (void)
	{
		float3 const bodyColor = { 0.1f, 0.1f, 0.9f };	// very bluish
		drawBasic3dSphericalVehicle( radius(), position(), forward(), side(), up(), bodyColor );
	}
};

class BoidsGroup : public AgentGroup
{
private:
	KNNData *				m_pKNNSelf;
	KNNData *				m_pKNNObstacles;

public:
	BoidsGroup( BoidsWorld * pWorld )
	:	AgentGroup( gWorldCells, g_knn ),
		m_pKNNSelf( NULL ),
		m_pKNNObstacles( NULL )
	{
		m_pKNNSelf = new KNNData( gEnemyCount, g_knn );
		m_pKNNObstacles = new KNNData( gEnemyCount, g_kno );

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
		std::vector< float3 > & positions = m_obstacleGroupData.hvPosition();
		std::vector< float > & radii = m_obstacleGroupData.hvRadius();

		float minClearance;
		const float requiredClearance = 2.0f; //gSeeker->radius() * 4; // 2 x diameter

		ObstacleData od;

		do
		{
			minClearance = FLT_MAX;

			od.radius = frandom2 (1.5f, 4.0f); // random radius between 1.5 and 4
			od.position = float3_scalar_multiply(float3_randomVectorOnUnitRadiusXZDisk(), g_fMaxStartRadius * 1.1f);

/*
			// Make sure it doesn't overlap with the home base.
			float d = float3_distance (od.position, make_float3( 0.f, 0.f, 0.f ) );
			float clearance = d - (od.radius + (g_fHomeBaseRadius - requiredClearance));
			if ( clearance < minClearance )
				minClearance = clearance;
*/

			// Make sure it doesn't overlap with any of the other obstacles.
			for( size_t i = 0; i < Size(); i++ )
			{
				float d = float3_distance( od.position, positions[i] );
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
			draw3dCircle( od.radius, od.position, OpenSteerDemo::camera.forward(), color, 20 );
		}
	}
};
//
//void randomizeHeading( VehicleData &vehicleData )
//{	
//	vehicleData.up = float3_up();
//	vehicleData.forward = float3_RandomUnitVectorOnXZPlane();
//	vehicleData.side = float3_LocalRotateForwardToSide(vehicleData.forward);
//}

void BoidsGroup::reset(void)
{
	Clear();
	//static unsigned int id = 0;
	// Add the required number of enemies.
	while(Size() < gEnemyCount)
	{
		BoidsBase enemy;
		VehicleData &edata = enemy.getVehicleData();
		VehicleConst &econst = enemy.getVehicleConst();

		edata.speed = 3.0f;
		econst.maxForce = 3.0f;
		econst.maxSpeed = 3.0f;
		randomizeStartingPositionAndHeadingBoids( edata.position, econst.radius, edata.up, edata.forward, edata.side );
		
		bool success = AddVehicle(edata, econst);
	}

	// Transfer the data to the device.
	SyncDevice();

	// Compute the initial KNN for this group with itself.
	// Update the KNN database for the group.
	updateKNNDatabase( this, g_pWorld->GetBinData() );
	findKNearestNeighbors( this, m_pKNNSelf, g_pWorld->GetBinData(), this, g_searchRadiusNeighbors );
}

void BoidsGroup::draw(void)
{
#if defined NO_DRAW
	return;
#endif

	// Draw all of the enemies
	float3 bodyColor = make_float3(0.6f, 0.4f, 0.4f); // redish

	VehicleConst vc;
	VehicleData vd;

	AgentGroupConst & m_agentGroupConst = g_pBoids->GetAgentGroupConst();
	AgentGroupData & m_agentGroupData = g_pBoids->GetAgentGroupData();

#if defined ANNOTATION_LINES || defined ANNOTATION_TEXT
	// Temporary storage used for annotation.
	uint KNNIndices[g_knn];
	float KNNDistances[g_knn];
#endif

	// For each enemy...
	for( size_t i = 0; i < g_pBoids->Size(); i++ )
	//for( size_t i = 0; i < g_pBoids->Size(); i += 100 )
	{
		// Get its varialbe and constant data.
		m_agentGroupConst.getVehicleData( i, vc );
		m_agentGroupData.getVehicleData( i, vd );

		// Draw the agent.
		drawBasic3dSphericalVehicle( vc.radius, vd.position, vd.forward, vd.side, vd.up, bodyColor );

#if defined ANNOTATION_LINES || defined ANNOTATION_TEXT
		//
		// Annotation
		//

		// Pull the KNN data for this agent from the nearest neighbor database.
		m_pKNNSelf->getAgentData( i, KNNIndices, KNNDistances );
		
#endif

#if defined ANNOTATION_TEXT
		// annotate the agent with useful data.
		const float3 textOrigin = float3_add( vd.position, make_float3( 0, 0.25, 0 ) );
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
		VehicleData vdOther;

		// Draw the KNN links.
		for( uint j = 0; j < g_knn; j++ )
		{
			if( KNNIndices[j] < g_pBoids->Size() )
			{
				m_agentGroupData.getVehicleData( KNNIndices[j], vdOther );
				drawLine( vd.position, vdOther.position, make_float3( 1.f, 1.f, 1.f ) );
			}
		}
#endif
	}
}

void BoidsGroup::update(const float currentTime, const float elapsedTime)
{
	// Update the positions in the KNNDatabase for the group.
	updateKNNDatabase( this, g_pWorld->GetBinData() );

	// Update the KNNDatabases
	//findKNearestNeighbors( this, m_pKNNObstacles, g_pWorld->GetBinData(), gObstacles, g_searchRadiusObstacles );
	findKNearestNeighbors( this, m_pKNNSelf, g_pWorld->GetBinData(), this, g_searchRadiusNeighbors );

	// Avoid collision with obstacles.
	//steerToAvoidObstacles( this, gObstacles, m_pKNNObstacles, g_fMinTimeToObstacle, g_fWeightObstacleAvoidance, g_maskObstacleAvoidance );

	// Avoid collision with self.
	//steerToAvoidNeighbors( this, m_pKNNSelf, this,  g_fMinTimeToCollision, g_fMinSeparationDistance, g_fWeightAvoidNeighbors, g_maskNeighborAvoidance );

	// Flocking.
	steerForSeparation( this, m_pKNNSelf, this, g_fMinFlockingDistance, g_fMaxSeparationDistance, g_fCosMaxFlockingAngle, g_fWeightSeparation, g_maskSeparation );
	steerForCohesion( this, m_pKNNSelf, this, g_fMinFlockingDistance, g_fMaxFlockingDistance, g_fCosMaxFlockingAngle, g_fWeightCohesion, g_maskCohesion );
	steerForAlignment( this, m_pKNNSelf, this, g_fMinFlockingDistance, g_fMaxFlockingDistance, g_fCosMaxFlockingAngle, g_fWeightAlignment, g_maskAlignment );

	// Pursue target.
	//steerForPursuit( this, gSeeker->getVehicleData(), g_fMaxPursuitPredictionTime, g_fWeightPursuit, g_maskPursuit );
	//steerForPursuit( this, g_pWanderer->getVehicleData(), g_fMaxPursuitPredictionTime, g_fWeightPursuit, g_maskPursuit );
	steerForSeek( this, g_pWanderer->position(), g_fWeightSeek, g_maskSeek );

	//steerForEvade( this, g_pWanderer->position(), g_pWanderer->forward(), g_pWanderer->speed(), g_fMaxPursuitPredictionTime, g_fWeightEvade, g_maskEvade );

	// Apply steering.
	updateGroup( this, elapsedTime );
}

// ----------------------------------------------------------------------------
// reset state
void BoidsBase::reset (void)
{
	//_data.id = serialNumber;
	_const.id = serialNumber;

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

		g_pWorld = new BoidsWorld( gWorldCells, gWorldSize, g_maxSearchRadius );

		//g_pKNNBinData = new KNNBinData( gWorldCells, gWorldSize, g_searchRadius );
		//g_pWallData = new wall_data;
		//g_pWallData->SplitWalls( g_pKNNBinData->hvCells() );

		//g_pObstacles = new BoidsObstacleGroup( gWorldCells, g_kno );
		//g_pObstacles->reset();

		int numDevices;
		cudaGetDeviceCount(&numDevices);

		// TODO: more intelligent selection of the CUDA device.

		CUDA_SAFE_CALL( cudaSetDevice( 0 ) );

        // create the specified number of enemies, 
        // storing pointers to them in an array.
		g_pBoids = new BoidsGroup( g_pWorld );

		g_pWanderer = new BoidsWanderer;
		all.push_back( g_pWanderer );

		/*
        // initialize camera
		OpenSteerDemo::init3dCamera( *g_pWanderer );
		OpenSteerDemo::camera.mode = Camera::cmFixedDistanceOffset;
        OpenSteerDemo::camera.fixedTarget = make_float3(15, 0, 0);
        OpenSteerDemo::camera.fixedPosition = make_float3(0, 0, 500);
		*/

		// initialize camera
        OpenSteerDemo::init3dCamera ( *g_pWanderer );
		OpenSteerDemo::camera.mode = Camera::cmFixedDistanceOffset;
        OpenSteerDemo::camera.fixedDistDistance = 100.f;
        OpenSteerDemo::camera.fixedDistVOffset = 1.f;
        OpenSteerDemo::camera.lookdownDistance = 20;
        OpenSteerDemo::camera.aimLeadTime = 0.5;
        OpenSteerDemo::camera.povOffset = make_float3( 0, 0.5, -2 );
    }

    void update (const float currentTime, const float elapsedTime)
    {
		// update the enemy group
		g_pBoids->update(currentTime, elapsedTime);

		g_pWanderer->update( currentTime, elapsedTime );
	}

    void redraw (const float currentTime, const float elapsedTime)
    {
		AbstractVehicle& selected = *g_pWanderer;

		// update camera
        OpenSteerDemo::updateCamera (currentTime, elapsedTime, selected);

		// draw the obstacles
		//g_pObstacles->draw();

        // draw the enemy
		g_pBoids->draw();

		// draw the world
		g_pWorld->draw();

		// display status in the upper left corner of the window
		std::ostringstream status;
		//status << std::left << std::setw( 25 ) << "No. obstacles: " << std::setw( 10 ) << g_pObstacles->Size() << std::endl;
		status << std::left << std::setw( 25 ) << "No. agents: " << std::setw( 10 ) << g_pBoids->Size() << std::endl;
		status << std::left << std::setw( 25 ) << "World dim: " << std::setw( 10 ) << gDim << std::endl;
		status << std::left << std::setw( 25 ) << "World cells: " << std::setw( 10 ) << gCells << std::endl;
		status << std::left << std::setw( 25 ) << "Search radius neighbors: " << std::setw( 10 ) << g_searchRadiusNeighbors << std::endl;
		status << std::left << std::setw( 25 ) << "Search radius obstacles: " << std::setw( 10 ) << g_searchRadiusObstacles << std::endl;
		status << std::left << std::setw( 25 ) << "Wanderer position: " << selected.position().x << ", " << selected.position().y << ", " << selected.position().z << std::endl;
		const float h = drawGetWindowHeight ();
		const float3 screenLocation = make_float3(10, h-50, 0);
		draw2dTextAt2dLocation (status, screenLocation, gGray80);
    }

    void close (void)
    {
		delete g_pBoids;

		//delete g_pObstacles;

		delete g_pWorld;

		delete g_pWanderer;
    }

    void reset (void)
    {
		g_pBoids->reset();

		//g_pObstacles->reset();

		g_pWanderer->reset();

        // reset camera position
		//OpenSteerDemo::camera.reset();

        // make camera jump immediately to new position
        OpenSteerDemo::camera.doNotSmoothNextMove ();
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


