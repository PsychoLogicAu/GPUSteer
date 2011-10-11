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
//#define ANNOTATION_WALL_LINES
//#define ANNOTATION_TEXT
//#define ANNOTATION_CELLS
//#define NO_DRAW
#define NO_DRAW_OBSTACLES
#define NO_DRAW_OUTSIDE_RANGE

// ----------------------------------------------------------------------------
// forward declarations
class CtfEnemyGroup;
class CtfSeeker;
class CtfBase;
class CtfObstacleGroup;

#define SAFE_DELETE( x )	{ if( x ){ delete x; x = NULL; } }

// ----------------------------------------------------------------------------
// globals


// Using cell diameter of 7

//const int	gEnemyCount					= 100;
//const float gDim						= 50;
//const int	gCells						= 10;

//const int gEnemyCount					= 1000;
//const float gDim						= 200;
//const int gCells						= 50;

//const int gEnemyCount					= 10000;
//const float gDim						= 635;
////const int gCells						= 91;
//const int gCells						= 200;

const int gEnemyCount					= 100000;
const float gDim						= 2000;
//const int gCells						= 285;
const int gCells						= 500;

//const int gEnemyCount					= 1000000;
//const float gDim						= 6350;
////const int gCells						= 907;
////const int gCells						= 1814;
//const int gCells						= 2048;



//const int gEnemyCount					= 1000;
//const float gDim						= 100;
//const int gCells						= 25;

const int gObstacleCount				= 1;

float const	g_fPathRadius				= 23.5f;

uint const	g_knn						= 5;		// Number of near neighbors to keep track of.
uint const	g_kno						= 2;		// Number of near obstacles to keep track of.
uint const	g_knw						= 3;		// Number of near walls to keep track of.
uint const	g_maxSearchRadius			= 1;		// Distance in cells for the maximum search radius.
uint const	g_searchRadiusNeighbors		= 1;		// Distance in cells to search for neighbors.
uint const	g_searchRadiusObstacles		= 1;		// Distance in cells to search for obstacles.
uint const	g_searchRadiusWalls			= 1;		// Distance in cells to search for obstacles.

float const g_fMaxPursuitPredictionTime	= 10.0f;		// Look-ahead time for pursuit.
float const g_fMinTimeToCollision		= 2.0f;		// Look-ahead time for neighbor avoidance.
float const g_fMinTimeToObstacle		= 5.0f;		// Look-ahead time for obstacle avoidance.
float const g_fMinTimeToWall			= 5.0f;		// Look-ahead time for wall avoidance.
float const g_fPathPredictionTime		= 5.f;		// Point on path in future to aim for.

float const g_fMinSeparationDistance	= 0.5f;		// Mini
float const g_fMaxSeparationDistance	= 1.3f;		// Maximum range for separation behavior.
float const g_fMinFlockingDistance		= 0.5f;
float const g_fMaxFlockingDistance		= 7.f;
float const g_fCosMaxFlockingAngle		= cosf(150 * (float)M_PI / 180.f);	// 350 degrees - used in "An efficient GPU implementation for large scale individual-based simulation of collective behavior"

// Weights for behaviors.
float const	g_fWeightAlignment			= 2.f;
float const	g_fWeightCohesion			= 1.f;
float const	g_fWeightSeparation			= 10.f;

float const g_fWeightPursuit			= 1.f;
float const g_fWeightSeek				= 1.f;

float const g_fWeightFollowPath			= 1.f;

float const g_fWeightObstacleAvoidance	= 1.f;
float const g_fWeightWallAvoidance		= 1.f;
float const g_fWeightAvoidNeighbors		= 1.f;

// Masks for behaviors.
uint const	g_maskAlignment				= KERNEL_AVOID_WALLS_BIT | KERNEL_SEPARATION_BIT;
uint const	g_maskCohesion				= KERNEL_AVOID_WALLS_BIT | KERNEL_SEPARATION_BIT;
uint const	g_maskSeparation			= KERNEL_AVOID_WALLS_BIT;

uint const	g_maskSeek					= KERNEL_AVOID_WALLS_BIT | KERNEL_SEPARATION_BIT;
uint const	g_maskFlee					= KERNEL_AVOID_OBSTACLES_BIT | KERNEL_AVOID_WALLS_BIT | KERNEL_AVOID_NEIGHBORS_BIT;
uint const	g_maskPursuit				= KERNEL_AVOID_OBSTACLES_BIT | KERNEL_AVOID_WALLS_BIT | KERNEL_AVOID_NEIGHBORS_BIT;
uint const	g_maskEvade					= KERNEL_AVOID_OBSTACLES_BIT | KERNEL_AVOID_WALLS_BIT | KERNEL_AVOID_NEIGHBORS_BIT;

uint const	g_maskFollowPath			= KERNEL_SEPARATION_BIT;

uint const	g_maskObstacleAvoidance		= 0;
uint const	g_maskNeighborAvoidance		= 0;
uint const	g_maskWallAvoidance			= 0;

uint const	g_maskAntiPenetrationAgents	= KERNEL_AVOID_WALLS_BIT;

const float3 gWorldSize					= make_float3( gDim, 10.f, gDim );
const uint3 gWorldCells					= make_uint3( gCells, 1, gCells );

// Start position.
float3 const g_f3StartBaseCenter		= make_float3( 0.f, 0.f, 0.25f * gWorldSize.z );
float const g_fMinStartRadius			= 0.0f;
float const g_fMaxStartRadius			= 0.20f * min( gWorldSize.x, gWorldSize.z );

// Goal position.
float3 const g_f3GoalPosition			= make_float3( 0.f, 0.f, -2.f * /*0.25f * */gWorldSize.z );

float3 const g_f3HomeBaseCenter			= make_float3( 0.f, 0.f, 0.f );
const float g_fHomeBaseRadius			= 1.5f;



//const float g_fMaxStartRadius				= 60.0f;


const float gBrakingRate				= 0.75f;

const float3 evadeColor					= make_float3(0.6f, 0.6f, 0.3f); // annotation
const float3 seekColor					= make_float3(0.3f, 0.6f, 0.6f); // annotation
const float3 clearPathColor				= make_float3(0.3f, 0.6f, 0.3f); // annotation

const float gAvoidancePredictTimeMin	= 0.9f;
const float gAvoidancePredictTimeMax	= 2.0f;
float gAvoidancePredictTime				= gAvoidancePredictTimeMin;

bool enableAttackSeek					= true; // for testing (perhaps retain for UI control?)
bool enableAttackEvade					= true; // for testing (perhaps retain for UI control?)

// count the number of times the simulation has reset (e.g. for overnight runs)
int resetCount = 0;

// Function prototypes.
void randomizeStartingPositionAndHeading( float4 & position, float const radius, float3 & up, float4 & forward, float3 & side );

class CtfWorld
{
private:
// Bin data to be used for KNN lookups.
	KNNBinData *							m_pKNNBinData;

	WallGroup *								m_pWallGroup;

public:
	CtfWorld( uint3 const& worldCells, float3 const& worldSize, uint const maxSearchRadius )
	:	m_pKNNBinData( NULL ),
		m_pWallGroup( NULL )
	{
		m_pKNNBinData = new KNNBinData( worldCells, worldSize, maxSearchRadius );
		m_pWallGroup = new WallGroup( worldCells, g_knw );

		// Load the walls from a file.
		m_pWallGroup->LoadFromFile( "10kAgentsChoke.map" );
		// Clip the walls to the edges of the cells.
		m_pWallGroup->SplitWalls( m_pKNNBinData->hvCells() );

		// Send the data to the device.
		m_pWallGroup->SyncDevice();

		// Update the KNN Database for the WallGroup.
		updateKNNDatabase( m_pWallGroup, m_pKNNBinData );
	}

	~CtfWorld( void )
	{
		SAFE_DELETE( m_pKNNBinData ); 
		SAFE_DELETE( m_pWallGroup );
	}

	WallGroup * GetWalls( void )	{ return m_pWallGroup; }

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
			/*
			0    1
			+----+
			|    |
			|    |
			+----+
			2    3
			*/
			float3 const p0		= { it->minBound.x, 0.f, it->maxBound.z };
			float3 const p1		= { it->maxBound.x, 0.f, it->maxBound.z };
			float3 const p2		= { it->minBound.x, 0.f, it->minBound.z };
			float3 const p3		= { it->maxBound.x, 0.f, it->minBound.z };

			drawLine( p0, p1, cellColor );
			drawLine( p0, p2, cellColor );
			drawLine( p1, p3, cellColor );
			drawLine( p2, p3, cellColor );
		}
#endif

		//
		//	Draw the walls.
		//

		// Get references to the vectors.
		std::vector< float4 > const& start		= m_pWallGroup->GetWallGroupData().hvLineStart();
		std::vector< float4 > const& mid		= m_pWallGroup->GetWallGroupData().hvLineMid();
		std::vector< float4 > const& end		= m_pWallGroup->GetWallGroupData().hvLineEnd();
		std::vector< float4 > const& normal		= m_pWallGroup->GetWallGroupData().hvLineNormal();

		float3 const lineColor = { 1.f, 0.f, 0.f };

		// For each line in the host data...
		for( uint i = 0; i < start.size(); i++ )
		{
			// Draw the line.
			drawLine( make_float3(start[i]), make_float3(end[i]), lineColor );
			// Draw the normal.
			drawLine( make_float3(mid[i]), float3_add( make_float3(mid[i]), make_float3(normal[i]) ), lineColor );
		}
	}

	KNNBinData * GetBinData( void )		{ return m_pKNNBinData; }
};


// ----------------------------------------------------------------------------
// state for OpenSteerDemo PlugIn
//
// XXX consider moving this inside CtfPlugIn
// XXX consider using STL (any advantage? consistency?)
CtfSeeker*			gSeeker;
CtfSeeker*			ctfSeeker	= NULL;
CtfEnemyGroup*		gEnemies;
CtfObstacleGroup*	gObstacles;
CtfWorld *			g_pWorld;

// ----------------------------------------------------------------------------
// This PlugIn uses two vehicle types: CtfSeeker and CtfEnemy.  They have a
// common base class: CtfBase which is a specialization of SimpleVehicle.
class CtfBase : public SimpleVehicle
{
public:
    // constructor
    CtfBase () {reset ();}

    // reset state
    void reset (void);

    // draw this character/vehicle into the scene
    void draw (void);

    // annotate when actively avoiding obstacles
    void annotateAvoidObstacle (const float minDistanceToCollision);

    void drawHomeBase (void);

    enum seekerState {running, tagged, atGoal};

    // for draw method
    float3 bodyColor;

    // xxx store steer sub-state for anotation
    bool avoiding;
};

class CtfSeeker : public CtfBase
{
public:
    // constructor
    CtfSeeker () {reset ();}

    // reset state
    void reset (void);

    // per frame simulation update
    void update (const float currentTime, const float elapsedTime);

    // is there a clear path to the goal?
    bool clearPathToGoal (void);

    float3 steeringForSeeker (float const elapsedTime);
    void updateState (const float currentTime);
    void draw (void);
    float3 steerToEvadeAllDefenders (void);
    float3 XXXsteerToEvadeAllDefenders (void);
    void adjustObstacleAvoidanceLookAhead (const bool clearPath);
    void clearPathAnnotation (const float threshold,
                              const float behindcThreshold,
                              const float3& goalDirection);

    seekerState state;
    bool evading; // xxx store steer sub-state for anotation
	bool wandering;
    float lastRunningTime; // for auto-reset
};

class CtfEnemyGroup : public AgentGroup
{
private:
	KNNData *				m_pKNNSelf;
	KNNData *				m_pKNNObstacles;
	KNNData *				m_pKNNWalls;

	PolylinePathwayCUDA *	m_pPath;

public:
	CtfEnemyGroup( CtfWorld * pWorld )
	:	AgentGroup( gWorldCells, g_knn ),
		m_pKNNSelf( NULL ),
		m_pKNNObstacles( NULL ),
		m_pKNNWalls( NULL )
	{
		m_pKNNSelf = new KNNData( gEnemyCount, g_knn );
		m_pKNNObstacles = new KNNData( gEnemyCount, g_kno );
		m_pKNNWalls = new KNNData( gEnemyCount, g_knw );

		// Create a path.
		std::vector< float3 > points;
		//points.push_back( g_f3StartBaseCenter );
		points.push_back( g_f3HomeBaseCenter );
		points.push_back( g_f3GoalPosition );
		m_pPath = new PolylinePathwayCUDA( points, g_fPathRadius, false );

		reset();
	}
	virtual ~CtfEnemyGroup(void)
	{
		SAFE_DELETE( m_pKNNSelf );
		SAFE_DELETE( m_pKNNObstacles );
		SAFE_DELETE( m_pKNNWalls );
		SAFE_DELETE( m_pPath );
	}

	void reset(void);
	void draw(void);

	void update(const float currentTime, const float elapsedTime);

	void randomizeStartingPositionAndHeading( AgentData & agentData );
	void randomizeHeadingOnXZPlane( AgentData & agentData );
};

#define testOneObstacleOverlap(radius, center)					\
{																\
    float d = float3_distance (od.position, center);			\
    float clearance = d - (od.radius + (radius));				\
    if (minClearance > clearance) minClearance = clearance;		\
}

class CtfObstacleGroup : public ObstacleGroup
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
			od.position = make_float4( float3_scalar_multiply(float3_randomVectorOnUnitRadiusXZDisk(), g_fMaxStartRadius * 1.1f), 0.f );

			// Make sure it doesn't overlap with any of the other obstacles.
			for( size_t i = 0; i < Size(); i++ )
			{
				float d = float3_distance( make_float3( od.position ), make_float3( positions[i] ) );
				float clearance = d - (od.radius + radii[i]);
				if ( clearance < minClearance )
					minClearance = clearance;
			}
		} while (minClearance < requiredClearance);

		// add new non-overlapping obstacle to registry
		AddObstacle( od );
	}

public:
	CtfObstacleGroup( uint3 const& worldCells, uint const kno )
	:	ObstacleGroup( worldCells, kno )
	{
	}
	virtual ~CtfObstacleGroup( void ) {}

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
#ifndef NO_DRAW_OBSTACLES
        const float3 color = make_float3(0.8f, 0.6f, 0.4f);

		for( uint i = 0; i < m_nCount; i++ )
		{
			ObstacleData od;
			GetDataForObstacle( i, od );

			drawXZCircle( od.radius, make_float3( od.position ), color, 20 );
		}
#endif
	}
};

void CtfEnemyGroup::randomizeHeadingOnXZPlane( AgentData &agentData )
{	
	agentData.up = float3_up();
	agentData.direction = make_float4( float3_RandomUnitVectorOnXZPlane(), 0.f );
	agentData.side = float3_LocalRotateForwardToSide(make_float3(agentData.direction));
}

void CtfEnemyGroup::reset(void)
{
	Clear();
	//static unsigned int id = 0;
	// Add the required number of enemies.
	while(Size() < gEnemyCount)
	{
		CtfBase enemy;
		AgentData &edata = enemy.getVehicleData();

		edata.speed = 3.0f;
		edata.maxForce = 3.0f;
		edata.maxSpeed = 3.0f;
		randomizeStartingPositionAndHeading( edata );
		
		bool success = AddAgent( edata );
	}

	// Transfer the data to the device.
	SyncDevice();

	// Compute the initial KNN for this group with itself.
	// Update the KNN database for the group.
	updateKNNDatabase( this, g_pWorld->GetBinData() );
	findKNearestNeighbors( this, m_pKNNSelf, g_pWorld->GetBinData(), this, g_searchRadiusNeighbors );
}

// ----------------------------------------------------------------------------
void CtfEnemyGroup::randomizeStartingPositionAndHeading( AgentData & aData )
{
    // randomize position on a ring between inner and outer radii
    // centered around the home base
    const float rRadius = frandom2 (g_fMinStartRadius, g_fMaxStartRadius);
    const float3 randomOnRing = float3_scalar_multiply(float3_RandomUnitVectorOnXZPlane (), rRadius);
	aData.position = make_float4( float3_add(g_f3StartBaseCenter, randomOnRing), 0.f );

    // are we are too close to an obstacle?
	//float distance;
	// TODO: implement distance check.
	//if(gObstacles->MinDistanceToObstacle(vehicleData.position, vehicleConst.radius * 5, distance))
 //   {
 //       // if so, retry the randomization (this recursive call may not return
 //       // if there is too little free space)
 //       randomizeStartingPositionAndHeading (vehicleData, vehicleConst);
 //   }
 //   else
 //   {
        // otherwise, if the position is OK, randomize 2D heading
		randomizeHeadingOnXZPlane(aData);
    //}
}

void CtfEnemyGroup::draw(void)
{
#if defined NO_DRAW
	return;
#endif


	// Draw all of the enemies
	float3 bodyColor = make_float3(0.6f, 0.4f, 0.4f); // redish

	AgentData ad;

	AgentGroupData & agd = gEnemies->GetAgentGroupData();

	// For each enemy...
	for( size_t i = 0; i < gEnemies->Size(); i++ )
	{
		// Get its varialbe and constant data.
		agd.getAgentData( i, ad );

#if defined NO_DRAW_OUTSIDE_RANGE
		if( float3_distanceSquared( make_float3( ad.position ), g_f3HomeBaseCenter ) > 2500.f )
			continue;
#endif

		// Draw the agent.
		drawBasic2dCircularVehicle( ad.radius, make_float3(ad.position), make_float3(ad.direction), ad.side, bodyColor );

#if defined ANNOTATION_LINES
		//
		// Annotation
		//

		// Temporary storage used for annotation.
		uint KNNIndices[g_knn];
		float KNNDistances[g_knn];

		// Pull the KNN data for this agent from the nearest neighbor database.
		m_pKNNSelf->getAgentData( i, KNNIndices, KNNDistances );
#endif
#if defined ANNOTATION_WALL_LINES
		// Temporary storage for the KNN data.
		uint KNWIndices[g_knw];
		float KNWDistances[g_knw];

		m_pKNNWalls->getAgentData( i, KNWIndices, KNWDistances );

		WallGroupData const& wgd = g_pWorld->GetWalls()->GetWallGroupData();

		std::vector< float4 > const& hvLineMid = wgd.hvLineMid();

		// Draw the KNW links.
		for( uint j = 0; j < g_knw; j++ )
		{
			if( KNWIndices[j] < wgd.size() )
			{
				float3 const& lineMid = make_float3(hvLineMid[KNWIndices[j]]);
				drawLine( make_float3(ad.position), lineMid, make_float3( 1.f, 1.f, 1.f ) );
			}
		}
#endif

#if defined ANNOTATION_TEXT
		// annotate the agent with useful data.
		const float3 textOrigin = float3_add( make_float3(ad.position), make_float3( 0, 0.25, 0 ) );
		std::ostringstream annote;

		// Write this agent's index.
		annote << i << std::endl;
		annote << std::ends;

		draw2dTextAt3dLocation (annote, textOrigin, gWhite);
#endif

#if defined ANNOTATION_LINES
		AgentData adOther;

		// Draw the KNN links.
		for( uint j = 0; j < g_knn; j++ )
		{
			if( KNNIndices[j] < gEnemies->Size() )
			{
				agd.getAgentData( KNNIndices[j], adOther );
				drawLine( make_float3(ad.position), make_float3(adOther.position), make_float3( 1.f, 1.f, 1.f ) );
			}
		}
#endif
	}
}

void CtfEnemyGroup::update(const float currentTime, const float elapsedTime)
{
	// Update the positions in the KNNDatabase for the group.
	updateKNNDatabase( this, g_pWorld->GetBinData() );

	// Update the KNNDatabases
	//findKNearestNeighbors( this, m_pKNNObstacles, g_pWorld->GetBinData(), gObstacles, g_searchRadiusObstacles );
	findKNearestNeighbors( this, m_pKNNSelf, g_pWorld->GetBinData(), this, g_searchRadiusNeighbors );
	findKNearestNeighbors( this, m_pKNNWalls, g_pWorld->GetBinData(), g_pWorld->GetWalls(), g_searchRadiusWalls );


	// Avoid collisions with walls.
	steerToAvoidWalls( this, m_pKNNWalls, g_pWorld->GetWalls(), g_fMinTimeToWall, g_fWeightWallAvoidance, g_maskWallAvoidance );

	// Avoid collision with obstacles.
	//steerToAvoidObstacles( this, gObstacles, m_pKNNObstacles, g_fMinTimeToObstacle, g_fWeightObstacleAvoidance, g_maskObstacleAvoidance );

	// Avoid collision with self.
	//steerToAvoidNeighbors( this, m_pKNNSelf, this,  g_fMinTimeToCollision, g_fMinSeparationDistance, g_fWeightAvoidNeighbors, g_maskNeighborAvoidance );

	// Pursue target.
	//steerForPursuit( this, gSeeker->getVehicleData(), g_fMaxPursuitPredictionTime, g_fWeightPursuit, g_maskPursuit );

	steerToFollowPath( this, m_pPath, g_fPathPredictionTime, g_fWeightFollowPath, g_maskFollowPath );
	steerForSeek( this, g_f3GoalPosition, g_fWeightSeek, g_maskSeek );

	// Flocking.
	steerForSeparation( this, m_pKNNSelf, this, g_fMinSeparationDistance, g_fMaxSeparationDistance, g_fCosMaxFlockingAngle, g_fWeightSeparation, g_maskSeparation );
	steerForAlignment( this, m_pKNNSelf, this, g_fMinFlockingDistance, g_fMaxFlockingDistance, g_fCosMaxFlockingAngle, g_fWeightAlignment, g_maskAlignment );
	//steerForCohesion( this, m_pKNNSelf, this, g_fMinFlockingDistance, g_fMaxFlockingDistance, g_fCosMaxFlockingAngle, g_fWeightCohesion, g_maskCohesion );


	// Apply steering.
	updateGroup( this, elapsedTime );

	// Force anti-penetration.
	//antiPenetrationWall( this, m_pKNNWalls, g_pWorld->GetWalls(), elapsedTime, 0 );
	antiPenetrationAgents( this, m_pKNNSelf, this, g_maskAntiPenetrationAgents );

	/*
{
    // determine upper bound for pursuit prediction time
    const float seekerToGoalDist = Vec3::distance (gHomeBaseCenter,
                                                   gSeeker->position());
    const float adjustedDistance = seekerToGoalDist - radius()-gHomeBaseRadius;
    const float seekerToGoalTime = ((adjustedDistance < 0 ) ?
                                    0 :
                                    (adjustedDistance/gSeeker->speed()));
    const float maxPredictionTime = seekerToGoalTime * 0.9f;

    // determine steering (pursuit, obstacle avoidance, or braking)
    Vec3 steer (0, 0, 0);
    if (gSeeker->state == running)
    {
        const Vec3 avoidance =
            steerToAvoidObstacles (gAvoidancePredictTimeMin,
                                   (ObstacleGroup&) allObstacles);

        // saved for annotation
        avoiding = (avoidance == Vec3::zero);

        if (avoiding)
            steer = steerForPursuit (*gSeeker, maxPredictionTime);
        else
            steer = avoidance;
    }
    else
    {
        applyBrakingForce (gBrakingRate, elapsedTime);
    }
    applySteeringForce (steer, elapsedTime);

    // annotation
    annotationVelocityAcceleration ();
    recordTrailVertex (currentTime, position());


    // detect and record interceptions ("tags") of seeker
    const float seekerToMeDist = Vec3::distance (position(), 
                                                 gSeeker->position());
    const float sumOfRadii = radius() + gSeeker->radius();
    if (seekerToMeDist < sumOfRadii)
    {
        if (gSeeker->state == running) gSeeker->state = tagged;

        // annotation:
        if (gSeeker->state == tagged)
        {
            const Vec3 color (0.8f, 0.5f, 0.5f);
            annotationXZDisk (sumOfRadii,
                        (position() + gSeeker->position()) / 2,
                        color,
                        20);
        }
    }
}
	*/
}

// ----------------------------------------------------------------------------
// reset state
void CtfBase::reset (void)
{
	//_data.id = serialNumber;
	_data.id = serialNumber;

    SimpleVehicle::reset ();  // reset the vehicle 

    setSpeed (3);             // speed along Forward direction.
    setMaxForce (3.0);        // steering force is clipped to this magnitude
    setMaxSpeed (3.0);        // velocity is clipped to this magnitude

    avoiding = false;         // not actively avoiding

	randomizeStartingPositionAndHeading( _data.position, _data.radius, _data.up, _data.direction, _data.side );  // new starting position

    clearTrailHistory ();     // prevent long streaks due to teleportation
}

void CtfSeeker::reset (void)
{
    CtfBase::reset ();
    bodyColor = make_float3(0.4f, 0.4f, 0.6f); // blueish
    gSeeker = this;
    state = running;
    evading = false;
	wandering = false;

	//setPosition(float3_scalar_multiply(position(), 2.0f));
}


//void CtfEnemy::reset (void)
//{
//    CtfBase::reset ();
//    bodyColor = make_float3(0.6f, 0.4f, 0.4f); // redish
//}


// ----------------------------------------------------------------------------
// draw this character/vehicle into the scene
void CtfBase::draw (void)
{
    drawBasic2dCircularVehicle (*this, bodyColor);
    //drawTrail ();
}



//void randomizeStartingPositionAndHeading(VehicleData &vehicleData, VehicleConst &vehicleConst)
void randomizeStartingPositionAndHeading( float4 & position, float const radius, float3 & up, float4 & forward, float3 & side )
{
    // randomize position on a ring between inner and outer radii
    // centered around the home base
    const float rRadius = frandom2 ( g_fMinStartRadius, g_fMaxStartRadius );
	float3 const randomOnRing = float3_scalar_multiply( float3_RandomUnitVectorOnXZPlane(), rRadius );

    position = make_float4( float3_add( g_f3StartBaseCenter, randomOnRing ), 0.f );

    // are we are too close to an obstacle?
	//float distance;
	// TODO: Implement this distance check.
	//if ( gObstacles->MinDistanceToObstacle( position, radius * 5, distance ) )
 //   {
 //       // if so, retry the randomization (this recursive call may not return
 //       // if there is too little free space)
	//	randomizeStartingPositionAndHeading( position, radius, up, forward, side );
 //   }
 //   else
 //   {
        // otherwise, if the position is OK, randomize 2D heading
	float3 newForward;
    randomizeHeadingOnXZPlane( up, newForward, side );
	forward = make_float4( newForward, 0.f );
    //}
}

// ----------------------------------------------------------------------------
//void CtfEnemy::update (const float currentTime, const float elapsedTime)
//{
//    // determine upper bound for pursuit prediction time
//	const float seekerToGoalDist = float3_distance(gHomeBaseCenter,
//                                                   gSeeker->position());
//    const float adjustedDistance = seekerToGoalDist - radius()-gHomeBaseRadius;
//    const float seekerToGoalTime = ((adjustedDistance < 0 ) ?
//                                    0 :
//                                    (adjustedDistance/gSeeker->speed()));
//    const float maxPredictionTime = seekerToGoalTime * 0.9f;
//
//    // determine steering (pursuit, obstacle avoidance, or braking)
//    float3 steer = make_float3(0, 0, 0);
//    if (gSeeker->state == running)
//    {
//        const float3 avoidance =
//            steerToAvoidObstacles (*this, gAvoidancePredictTimeMin,
//                                   (ObstacleGroup&) allObstacles);
//
//        // saved for annotation
//        avoiding = float3_equals(avoidance, float3_zero());
//
//        if (avoiding)
//            steer = steerForPursuit (*this, *gSeeker, maxPredictionTime);
//        else
//            steer = avoidance;
//    }
//    else
//    {
//        applyBrakingForce (gBrakingRate, elapsedTime);
//    }
//    applySteeringForce (steer, elapsedTime);
//
//    // annotation
//    annotationVelocityAcceleration ();
//    recordTrailVertex (currentTime, position());
//
//
//    // detect and record interceptions ("tags") of seeker
//    const float seekerToMeDist = float3_distance (position(), 
//                                                 gSeeker->position());
//    const float sumOfRadii = radius() + gSeeker->radius();
//    if (seekerToMeDist < sumOfRadii)
//    {
//        if (gSeeker->state == running) gSeeker->state = tagged;
//
//        // annotation:
//        if (gSeeker->state == tagged)
//        {
//            const float3 color = make_float3(0.8f, 0.5f, 0.5f);
//            annotationXZDisk (sumOfRadii,
//						float3_scalar_divide(float3_add(position(), gSeeker->position()), 2),
//                        color,
//                        20);
//        }
//    }
//}


// ----------------------------------------------------------------------------
// are there any enemies along the corridor between us and the goal?
bool CtfSeeker::clearPathToGoal (void)
{
	return true;

    const float sideThreshold = radius() * 8.0f;
    const float behindThreshold = radius() * 2.0f;

    const float3 goalOffset = float3_subtract(g_f3HomeBaseCenter, make_float3(position()));
    const float goalDistance = float3_length(goalOffset);
    const float3 goalDirection = float3_scalar_divide(goalOffset, goalDistance);

    const bool goalIsAside = isAside (*this, g_f3HomeBaseCenter, 0.5);

    // for annotation: loop over all and save result, instead of early return 
    bool xxxReturn = true;

	AgentData aData;

	AgentGroupData & agd = gEnemies->GetAgentGroupData();

	// For each enemy...
	for( size_t i = 0; i < gEnemies->Size(); i++ )
	{
		// Get its data and const.
		agd.getAgentData( i, aData );

		const float eDistance = float3_distance ( make_float3(position()), make_float3(aData.position) );

		/*
		// Check if we were tagged.
		if(eDistance < (econst.radius + radius()))
		{
			state = tagged;
			return true;
		}
		*/

		const float timeEstimate = 0.3f * eDistance / aData.speed;
		const float3 eFuture = aData.predictFuturePosition(timeEstimate);
		const float3 eOffset = float3_subtract(eFuture, make_float3(position()));
		const float alongCorridor = float3_dot(goalDirection, eOffset);
        const bool inCorridor = ((alongCorridor > -behindThreshold) && (alongCorridor < goalDistance));
        const float eForwardDistance = float3_dot(make_float3(aData.direction), eOffset);

        // xxx temp move this up before the conditionals
		annotationXZCircle (aData.radius, eFuture, clearPathColor, 20); //xxx

        // consider as potential blocker if within the corridor
        if (inCorridor)
        {
            const float3 perp = float3_subtract(eOffset, float3_scalar_multiply(goalDirection, alongCorridor));
            const float acrossCorridor = float3_length(perp);
            if (acrossCorridor < sideThreshold)
            {
                // not a blocker if behind us and we are perp to corridor
				const float eFront = eForwardDistance + aData.radius;

                //annotationLine (position, forward*eFront, gGreen); // xxx
                //annotationLine (e.position, forward*eFront, gGreen); // xxx

                // xxx
                // std::ostringstream message;
                // message << "eFront = " << std::setprecision(2)
                //         << std::setiosflags(std::ios::fixed) << eFront << std::ends;
                // draw2dTextAt3dLocation (*message.str(), eFuture, gWhite);

                const bool eIsBehind = eFront < -behindThreshold;
                const bool eIsWayBehind = eFront < (-2 * behindThreshold);
                const bool safeToTurnTowardsGoal =
                    ((eIsBehind && goalIsAside) || eIsWayBehind);

                if (! safeToTurnTowardsGoal)
                {
                    // this enemy blocks the path to the goal, so return false
					annotationLine (make_float3(position()), make_float3(aData.position), clearPathColor);
                    // return false;
                    xxxReturn = false;
                }
            }
        }
    }

    // no enemies found along path, return true to indicate path is clear
    // clearPathAnnotation (sideThreshold, behindThreshold, goalDirection);
    // return true;
    if (xxxReturn)
        clearPathAnnotation (sideThreshold, behindThreshold, goalDirection);
    return xxxReturn;
}


// ----------------------------------------------------------------------------
void CtfSeeker::clearPathAnnotation (const float sideThreshold,
                                     const float behindThreshold,
                                     const float3& goalDirection)
{
    const float3 behindSide = float3_scalar_multiply(side(), sideThreshold);
	const float3 behindBack = float3_scalar_multiply(make_float3(forward()), -behindThreshold);
    const float3 pbb = float3_add(make_float3(position()), behindBack);
    const float3 gun = localRotateForwardToSide (goalDirection);
	const float3 gn = float3_scalar_multiply(gun, sideThreshold);
    const float3 hbc = g_f3HomeBaseCenter;
    annotationLine (float3_add(pbb, gn), float3_add(hbc, gn), clearPathColor);
    annotationLine (float3_subtract(pbb, gn), float3_subtract(hbc, gn), clearPathColor);
    annotationLine (float3_subtract(hbc, gn), float3_add(hbc, gn), clearPathColor);
    annotationLine (float3_subtract(pbb, behindSide), float3_add(pbb, behindSide), clearPathColor);
}


// ----------------------------------------------------------------------------
// xxx perhaps this should be a call to a general purpose annotation
// xxx for "local xxx axis aligned box in XZ plane" -- same code in in
// xxx Pedestrian.cpp


void CtfBase::annotateAvoidObstacle (const float minDistanceToCollision)
{
	const float3 boxSide = float3_scalar_multiply(side(), radius());
	const float3 boxFront = float3_scalar_multiply(make_float3(forward()), minDistanceToCollision);
	const float3 FR = float3_subtract(float3_add(make_float3(position()), boxFront), boxSide);
    const float3 FL = float3_add(float3_add(make_float3(position()), boxFront), boxSide);
    const float3 BR = float3_subtract(make_float3(position()), boxSide);
    const float3 BL = float3_add(make_float3(position()), boxSide);
    const float3 white = make_float3(1,1,1);
    annotationLine (FR, FL, white);
    annotationLine (FL, BL, white);
    annotationLine (BL, BR, white);
    annotationLine (BR, FR, white);
}


// ----------------------------------------------------------------------------


float3 CtfSeeker::steerToEvadeAllDefenders (void)
{
    float3 evade = float3_zero();

    const float goalDistance = float3_distance(g_f3HomeBaseCenter, make_float3(position()));

	AgentData aData;

	AgentGroupData & agd = gEnemies->GetAgentGroupData();

	// For each enemy...
	for( size_t i = 0; i < gEnemies->Size(); i++ )
	{
		// Get its data and const.
		agd.getAgentData( i, aData );

        const float3 eOffset = float3_subtract(make_float3(aData.position), make_float3(position()));
        const float eDistance = float3_length(eOffset);

        const float eForwardDistance = float3_dot(make_float3(forward()), eOffset);
        const float behindThreshold = radius() * 2;
        const bool behind = eForwardDistance < behindThreshold;
        if ((!behind) && (eDistance < 20))
        {
            if (eDistance < (goalDistance * 1.2)) //xxx
            {
				//float lookAheadTime = float3_length(toTarget) / (SPEED(offset) + target->speed);
                //const float timeEstimate = 0.5f * eDistance / edata.speed;//xxx
				const float timeEstimate = eDistance / (aData.speed + maxSpeed());//xxx
                //const float timeEstimate = 0.15f * eDistance / e.speed();//xxx
                const float3 future = aData.predictFuturePosition (timeEstimate);

                annotationXZCircle (aData.radius, future, evadeColor, 20); // xxx

                const float3 offset = float3_subtract(future, make_float3(position()));
                const float3 lateral = float3_perpendicularComponent(offset, make_float3(forward()));
                const float d = float3_length(lateral);
                const float weight = -1000 / (d * d);
				evade = float3_add(evade, float3_scalar_multiply(float3_scalar_divide(lateral, d), weight));
            }
        }
    }
    return evade;
}

// ----------------------------------------------------------------------------

float3 CtfSeeker::steeringForSeeker (float const elapsedTime)
{
    // determine if obstacle avodiance is needed
    const bool clearPath = clearPathToGoal();
    adjustObstacleAvoidanceLookAhead(clearPath);
	const float3 obstacleAvoidance = steerToAvoidObstacles(*this, gAvoidancePredictTime, *gObstacles);
	
	float3 const& seekerPosition = make_float3( position() );
	float const halfDim = 0.75f * gDim;

    // saved for annotation
    avoiding = ! float3_equals( obstacleAvoidance, float3_zero() );

    if (avoiding)
    {
        // use pure obstacle avoidance if needed
        return obstacleAvoidance;
    }
	else if(	seekerPosition.x < -halfDim		||
				seekerPosition.y < -halfDim		||
				seekerPosition.z < -halfDim		||

				seekerPosition.x > halfDim		||
				seekerPosition.y > halfDim		||
				seekerPosition.z > halfDim
		)
	{
		// Seeker has strayed outside of the world bounds. Seek back in.
		float3 seek = steerForSeek( *this, g_f3HomeBaseCenter );
		seek.y = 0.f;
		wandering = false;
		return seek;

	}
	else
	{
		AgentData ad;
		gEnemies->GetDataForAgent( gEnemies->Size(), ad );
		float3 seek = steerForSeek( *this, make_float3( ad.position ) );
		return seek;

		//float3 wander = steerForWander( *this, elapsedTime );
		//wander.y = 0.f;
		//wandering = true;
		//return wander;
	}
	/*
    else
    {
        // otherwise seek home base and perhaps evade defenders
        const float3 seek = xxxsteerForSeek (*this, gHomeBaseCenter);
        if (clearPath)
        {
            // we have a clear path (defender-free corridor), use pure seek

            // xxx experiment 9-16-02
            float3 s = limitMaxDeviationAngle (seek, 0.707f, forward());

            annotationLine (position(), float3_add(position(), float3_scalar_multiply(s, 0.2f)), seekColor);
            return s;
        }
        else
        {
            if (1) // xxx testing new evade code xxx
            {
                // combine seek and (forward facing portion of) evasion
                //const float3 evade = steerToEvadeAllDefenders ();
				const float3 evade = make_float3(0.0f, 0.0f, 0.0f);
                const float3 steer = float3_add(seek, limitMaxDeviationAngle (evade, 0.5f, forward()));

                // annotation: show evasion steering force
                annotationLine (position(), float3_add(position(), float3_scalar_multiply(steer, 0.2f)), evadeColor);
                return steer;
            }
            else
            {
                const float3 evade = XXXsteerToEvadeAllDefenders ();
                const float3 steer = limitMaxDeviationAngle (float3_add(seek, evade),
                                                           0.707f, forward());

                annotationLine (position(),float3_add(position(),seek), gRed);
                annotationLine (position(),float3_add(position(),evade), gGreen);

                // annotation: show evasion steering force
                annotationLine (position(),float3_add(position(),float3_scalar_multiply(steer,0.2f)),evadeColor);
                return steer;
            }
        }
    }
	*/
}


// ----------------------------------------------------------------------------
// adjust obstacle avoidance look ahead time: make it large when we are far
// from the goal and heading directly towards it, make it small otherwise.

void CtfSeeker::adjustObstacleAvoidanceLookAhead (const bool clearPath)
{
    if (clearPath)
    {
        evading = false;
        const float goalDistance = float3_distance(g_f3HomeBaseCenter,make_float3(position()));
        const bool headingTowardGoal = isAhead (*this, g_f3HomeBaseCenter, 0.98f);
        const bool isNear = (goalDistance/speed()) < gAvoidancePredictTimeMax;
        const bool useMax = headingTowardGoal && !isNear;
        gAvoidancePredictTime = (useMax ? gAvoidancePredictTimeMax : gAvoidancePredictTimeMin);
    }
    else
    {
        evading = true;
        gAvoidancePredictTime = gAvoidancePredictTimeMin;
    }
}


// ----------------------------------------------------------------------------
void CtfSeeker::updateState (const float currentTime)
{
    // if we reach the goal before being tagged, switch to atGoal state
    if (state == running)
    {
        const float baseDistance = float3_distance(make_float3(position()),g_f3HomeBaseCenter);
        if (baseDistance < (radius() + g_fHomeBaseRadius))
			state = atGoal;
    }

    // update lastRunningTime (holds off reset time)
    if (state == running)
    {
        lastRunningTime = currentTime;
    }
	else
    {
        const float resetDelay = 4;
        const float resetTime = lastRunningTime + resetDelay;
        if (currentTime > resetTime) 
        {
            // xxx a royal hack (should do this internal to CTF):
            OpenSteerDemo::queueDelayedResetPlugInXXX ();
        }
    }
}


// ----------------------------------------------------------------------------
void CtfSeeker::draw (void)
{
	/*
    // first call the draw method in the base class
    CtfBase::draw();

    // select string describing current seeker state
    char* seekerStateString = "";
    switch (state)
    {
    case running:
        if (avoiding)
            seekerStateString = "avoid obstacle";
        else if (evading)
            seekerStateString = "seek and evade";
		else if (wandering)
			seekerStateString = "wander";
        else
            seekerStateString = "seek goal";
        break;
    case tagged: seekerStateString = "tagged"; break;
    case atGoal: seekerStateString = "reached goal"; break;
    }

    // annote seeker with its state as text
    const float3 textOrigin = float3_add(make_float3(position()), make_float3(0, 0.25, 0));
    std::ostringstream annote;
    annote << seekerStateString << std::endl;
    annote << std::setprecision(2) << std::setiosflags(std::ios::fixed)
           << speed() << std::ends;
    draw2dTextAt3dLocation (annote, textOrigin, gWhite);
*/

    // display status in the upper left corner of the window
    std::ostringstream status;
    //status << seekerStateString << std::endl;
	status << std::left << std::setw( 25 ) << "No. obstacles: " << std::setw( 10 ) << gObstacles->Size() << std::endl;
	status << std::left << std::setw( 25 ) << "No. agents: " << std::setw( 10 ) << gEnemies->Size() << std::endl;
	status << std::left << std::setw( 25 ) << "World dim: " << std::setw( 10 ) << gDim << std::endl;
	status << std::left << std::setw( 25 ) << "World cells: " << std::setw( 10 ) << gCells << std::endl;
	status << std::left << std::setw( 25 ) << "Search radius neighbors: " << std::setw( 10 ) << g_searchRadiusNeighbors << std::endl;
	status << std::left << std::setw( 25 ) << "Search radius obstacles: " << std::setw( 10 ) << g_searchRadiusObstacles << std::endl;
	status << std::left << std::setw( 25 ) << "Position: " << position().x << ", " << position().z << std::endl;
	status << std::left << std::setw( 25 ) << "Reset count: " << std::setw( 10 ) << resetCount << std::ends;
    const float h = drawGetWindowHeight ();
    const float3 screenLocation = make_float3(10, h-50, 0);
    draw2dTextAt2dLocation (status, screenLocation, gGray80);
}


// ----------------------------------------------------------------------------
// update method for goal seeker
void CtfSeeker::update (const float currentTime, const float elapsedTime)
{
    // do behavioral state transitions, as needed
    updateState (currentTime);

    // determine and apply steering/braking forces
    float3 steer = make_float3(0, 0, 0);
    if (state == running)
    {
        steer = steeringForSeeker (elapsedTime);
    }
    else
    {
        applyBrakingForce (gBrakingRate, elapsedTime);
    }
    applySteeringForce (steer, elapsedTime);

    // annotation
    annotationVelocityAcceleration ();
    recordTrailVertex (currentTime, make_float3(position()));
}

// ----------------------------------------------------------------------------
// PlugIn for OpenSteerDemo


class CtfPlugIn : public PlugIn
{
private:

public:

    const char* name (void) {return "Choke Point";}

    float selectionOrderSortKey (void) {return 0.01f;}

    virtual ~CtfPlugIn() {} // be more "nice" to avoid a compiler warning

    void open (void)
    {
		OpenSteerDemo::setAnnotationOff();

		g_pWorld = new CtfWorld( gWorldCells, gWorldSize, g_maxSearchRadius );

		//g_pKNNBinData = new KNNBinData( gWorldCells, gWorldSize, g_searchRadius );
		//g_pWallData = new wall_data;
		//g_pWallData->SplitWalls( g_pKNNBinData->hvCells() );

		gObstacles = new CtfObstacleGroup( gWorldCells, g_kno );
		gObstacles->reset();

		int numDevices;
		cudaGetDeviceCount(&numDevices);

		// TODO: more intelligent selection of the CUDA device.

		CUDA_SAFE_CALL( cudaSetDevice( 0 ) );

        // create the seeker ("hero"/"attacker")
        ctfSeeker = new CtfSeeker;
        all.push_back (ctfSeeker);

        // create the specified number of enemies, 
        // storing pointers to them in an array.
		gEnemies = new CtfEnemyGroup( g_pWorld );
		//for (int i = 0; i < gEnemyCount; i++)
  //      {
		//	CtfBase *enemy = new CtfBase;
		//	enemy->reset();
		//	//TODO: need to initialise the data here.

		//	gEnemies->AddVehicle(enemy->serialNumber, enemy->getVehicleData(), enemy->getVehicleConst());
  //          //ctfEnemies[i] = new CtfEnemy;
  //          //all.push_back (ctfEnemies[i]);
  //      }

        // initialize camera
        OpenSteerDemo::init2dCamera (*ctfSeeker);
        OpenSteerDemo::camera.mode = Camera::cmFixedDistanceOffset;
        OpenSteerDemo::camera.fixedTarget = make_float3(15, 0, 0);
        OpenSteerDemo::camera.fixedPosition = make_float3(80, 60, 0);
    }

    void update (const float currentTime, const float elapsedTime)
    {
        //// update each enemy
        //for (int i = 0; i < ctfEnemyCount; i++)
        //{
        //    ctfEnemies[i]->update (currentTime, elapsedTime);
        //}

		// update the enemy group
		gEnemies->update(currentTime, elapsedTime);

        // update the seeker
        ctfSeeker->update (currentTime, elapsedTime);
	}

    void redraw (const float currentTime, const float elapsedTime)
    {
        // selected vehicle (user can mouse click to select another)
        AbstractVehicle& selected = *OpenSteerDemo::selectedVehicle;

        // vehicle nearest mouse (to be highlighted)
        AbstractVehicle& nearMouse = *OpenSteerDemo::vehicleNearestToMouse ();

        // update camera
        OpenSteerDemo::updateCamera (currentTime, elapsedTime, selected);

        // draw "ground plane" centered between base and selected vehicle
		const float3 goalOffset = float3_subtract(g_f3HomeBaseCenter, make_float3(OpenSteerDemo::camera.position()));
        const float3 goalDirection = float3_normalize(goalOffset);
        const float3 cameraForward = make_float3(OpenSteerDemo::camera.xxxls().forward());
        const float goalDot = float3_dot(cameraForward, goalDirection);
        const float blend = remapIntervalClip (goalDot, 1, 0, 0.5, 0);
        const float3 gridCenter = interpolate (blend,
                                             make_float3(selected.position()),
                                             g_f3HomeBaseCenter);
        OpenSteerDemo::gridUtility (gridCenter);

        // draw the seeker, obstacles and home base
        ctfSeeker->draw();
        //drawObstacles ();
		gObstacles->draw();
        drawHomeBase();

        // draw each enemy
        //for (int i = 0; i < ctfEnemyCount; i++) ctfEnemies[i]->draw ();
		gEnemies->draw();

		g_pWorld->draw();

        // highlight vehicle nearest mouse
        OpenSteerDemo::highlightVehicleUtility (nearMouse);
    }

    void close (void)
    {
        // delete seeker
        delete (ctfSeeker);
        ctfSeeker = NULL;

		delete gEnemies;

		delete gObstacles;

		//delete g_pKNNBinData;
		delete g_pWorld;

        //// delete each enemy
        //for (int i = 0; i < ctfEnemyCount; i++)
        //{
        //    delete (ctfEnemies[i]);
        //    ctfEnemies[i] = NULL;
        //}

        // clear the group of all vehicles
        all.clear();
    }

    void reset (void)
    {
        // count resets
        resetCount++;

        // reset the seeker ("hero"/"attacker")
        ctfSeeker->reset ();

		gEnemies->reset();

		gObstacles->reset();

		//AgentGroupData & vgd = gEnemies->GetAgentGroupData();
		//AgentGroupConst & vgc = gEnemies->GetAgentGroupConst();

		//// reset the enemies

		//// For each enemy...
		//for( size_t i = 0; i < gEnemies->Size(); i++ )
		//{
		//	//enemiesData[i].speed = 3.0f;
		//	vgd.hvSpeed()[i] = 3.0f;
		//	//randomizeHeadingOnXZPlane(enemiesData[i]);
		//	randomizeHeadingOnXZPlane( vgd.hvUp()[i], vgd.hvForward()[i], vgd.hvSide()[i] );
		//	randomizeStartingPositionAndHeading( vgd.hvPosition()[i], vgc.hvRadius()[i], vgd.hvUp()[i], vgd.hvForward()[i], vgd.hvSide()[i] );
		//}

        // reset camera position
        OpenSteerDemo::position2dCamera (*ctfSeeker);

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

    void drawHomeBase (void)
    {
        const float3 up = make_float3(0, 0.01f, 0);
        const float3 atColor = make_float3(0.3f, 0.3f, 0.5f);
        const float3 noColor = gGray50;
        const bool reached = ctfSeeker->state == CtfSeeker::atGoal;
        const float3 baseColor = (reached ? atColor : noColor);
        drawXZDisk (g_fHomeBaseRadius,    g_f3HomeBaseCenter, baseColor, 20);
        drawXZDisk (g_fHomeBaseRadius/15, float3_add(g_f3HomeBaseCenter, up), gBlack, 20);
    }

  //  void drawObstacles (void)
  //  {
  //      const float3 color = make_float3(0.8f, 0.6f, 0.4f);

		//for( uint i = 0; i < gObstacles->Size(); i++ )
		//{
		//	ObstacleData od;
		//	gObstacles->GetDataForObstacle( i, od );

		//	drawXZCircle( od.radius, od.position, color, 20 );
		//}
  //  }

    // a group (STL vector) of all vehicles in the PlugIn
    std::vector<CtfBase*> all;
};


CtfPlugIn gCtfPlugIn;


// ----------------------------------------------------------------------------


