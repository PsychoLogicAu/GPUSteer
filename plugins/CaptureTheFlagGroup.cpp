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

//#include "OpenSteer/CUDA/CUDAGlobals.cuh"

#include "OpenSteer/AgentData.h"

//#include "OpenSteer/CUDA/DebugUtils.h"

// Include the required KNN headers.
#include "OpenSteer/CUDA/KNNBinData.cuh"
#include "OpenSteer/CUDA/KNNBinningCUDA.cuh"

#include "OpenSteer/CUDA/WallData.cuh"
 
using namespace OpenSteer;

#ifndef min
#define min(a,b)            (((a) < (b)) ? (a) : (b))
#endif

//#define ANNOTATION_LINES
//#define ANNOTATION_TEXT
#define ANNOTATION_CELLS	// TODO: Draw the cells when this is on.
#define NO_DRAW

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
//const float	gDim						= 63;
//const int	gCells						= 9;

//const int gEnemyCount					= 1000;
//const float gDim						= 200;
//const int gCells						= 28;

//const int gEnemyCount					= 10000;
//const float gDim						= 635;
//const int gCells						= 91;

//const int gEnemyCount					= 100000;
//const float gDim						= 2000;
//const int gCells						= 285;

const int gEnemyCount					= 1000000;
const float gDim						= 6350;
//const int gCells						= 907;
const int gCells						= 1814;



//const int gEnemyCount					= 1000;
//const float gDim						= 100;
//const int gCells						= 25;

const int gObstacleCount				= 100;

uint const	g_knn						= 5;		// Number of near neighbors to keep track of.
uint const	g_kno						= 2;		// Number of near obstacles to keep track of.
uint const	g_maxSearchRadius			= 1;		// Distance in cells for the maximum search radius.
uint const	g_searchRadiusNeighbors		= 1;		// Distance in cells to search for neighbors.
uint const	g_searchRadiusObstacles		= 2;		// Distance in cells to search for obstacles.

float const g_fMaxPursuitPredictionTime	= 3.0f;		// Look-ahead time for pursuit.
float const g_fMinSeparationDistance	= 0.5f;		// Agents will steer hard to avoid other agents within this radius, and brake if other agent is ahead.
float const g_fMinTimeToCollision		= 1.0f;		// Look-ahead time for neighbor avoidance.
float const g_fMinTimeToObstacle		= 3.0f;		// Look-ahead time for obstacle avoidance.

// Weights for behaviors.
float const g_fWeightSeparation			= 0.5f;
float const g_fWeightPursuit			= 1.f;
float const g_fWeightObstacleAvoidance	= 1.f;
float const g_fWeightAvoidNeighbors		= 1.f;


const float3	gHomeBaseCenter			= make_float3(0, 0, 0);
const float		g_fHomeBaseRadius		= 1.5f;

const float3 gWorldSize					= make_float3( gDim, 10.f, gDim );
const uint3 gWorldCells					= make_uint3( gCells, 1, gCells );

const float gMinStartRadius				= 30.0f;
//const float gMaxStartRadius				= 60.0f;
const float gMaxStartRadius				= 0.5f * min( gWorldSize.x, gWorldSize.z );

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
void randomizeStartingPositionAndHeading( float3 & position, float const radius, float3 & up, float3 & forward, float3 & side );

class CtfWorld
{
private:
// Bin data to be used for KNN lookups.
	KNNBinData *							m_pKNNBinData;
	WallData *								m_pWallData;

public:
	CtfWorld( uint3 const& worldCells, float3 const& worldSize, uint const maxSearchRadius )
	:	m_pKNNBinData( NULL ),
		m_pWallData( NULL )
	{
		m_pKNNBinData = new KNNBinData( worldCells, worldSize, maxSearchRadius );
		m_pWallData = new WallData;

		m_pWallData->SplitWalls( m_pKNNBinData->hvCells() );
	}

	~CtfWorld( void )
	{
		SAFE_DELETE( m_pKNNBinData ); 
		SAFE_DELETE( m_pWallData );
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
	KNNData * m_pKNNSelf;
	KNNData * m_pKNNObstacles;

public:
	CtfEnemyGroup(void)
	:	AgentGroup( gWorldCells, g_knn ),
		m_pKNNSelf( NULL ),
		m_pKNNObstacles( NULL )
	{
		m_pKNNSelf = new KNNData( gEnemyCount, g_knn );
		//m_pKNNObstacles = new KNNData( gObstacleCount, g_kno );
		m_pKNNObstacles = new KNNData( gEnemyCount, g_kno );
		reset();
	}
	virtual ~CtfEnemyGroup(void)
	{
		delete m_pKNNSelf; m_pKNNSelf = NULL;
		delete m_pKNNObstacles; m_pKNNObstacles = NULL;
	}

	void reset(void);
	void draw(void);

	void update(const float currentTime, const float elapsedTime);

	void randomizeStartingPositionAndHeading(VehicleData &vehicleData, VehicleConst &vehicleConst);
	void randomizeHeadingOnXZPlane(VehicleData &vehicleData);
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
		std::vector< float3 > & positions = m_obstacleGroupData.hvPosition();
		std::vector< float > & radii = m_obstacleGroupData.hvRadius();

		float minClearance;
		const float requiredClearance = 2.0f; //gSeeker->radius() * 4; // 2 x diameter

		ObstacleData od;

		do
		{
			minClearance = FLT_MAX;

			od.radius = frandom2 (1.5f, 4.0f); // random radius between 1.5 and 4
			od.position = float3_scalar_multiply(float3_randomVectorOnUnitRadiusXZDisk(), gMaxStartRadius * 1.1f);


			// Make sure it doesn't overlap with the home base.
			float d = float3_distance (od.position, gHomeBaseCenter);
			float clearance = d - (od.radius + (g_fHomeBaseRadius - requiredClearance));
			if ( clearance < minClearance )
				minClearance = clearance;

			for( size_t i = 0; i < Size(); i++ )
			{
				d = float3_distance( od.position, positions[i] );
				clearance = d - (od.radius + radii[i]);
				if ( clearance < minClearance )
					minClearance = clearance;
			}
		} while (minClearance < requiredClearance);

		// add new non-overlapping obstacle to registry
		AddObstacle( od );
	}

public://gWorldCells, g_kno
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
		KNNBinningUpdateDBCUDA kernel( this, g_pWorld->GetBinData() );
		kernel.init();
		kernel.run();
		kernel.close();
	}

	void draw(void)
	{
        const float3 color = make_float3(0.8f, 0.6f, 0.4f);

		for( uint i = 0; i < m_nCount; i++ )
		{
			ObstacleData od;
			GetDataForObstacle( i, od );

			drawXZCircle( od.radius, od.position, color, 20 );
		}
	}
};

void CtfEnemyGroup::randomizeHeadingOnXZPlane(VehicleData &vehicleData)
{	
	vehicleData.up = float3_up();
	vehicleData.forward = float3_RandomUnitVectorOnXZPlane();
	vehicleData.side = float3_LocalRotateForwardToSide(vehicleData.forward);
}

void CtfEnemyGroup::reset(void)
{
	Clear();
	//static unsigned int id = 0;
	// Add the required number of enemies.
	while(Size() < gEnemyCount)
	{
		CtfBase enemy;
		VehicleData &edata = enemy.getVehicleData();
		VehicleConst &econst = enemy.getVehicleConst();

		edata.speed = 3.0f;
		econst.maxForce = 3.0f;
		econst.maxSpeed = 3.0f;
		randomizeStartingPositionAndHeading(edata, econst);
		
		bool success = AddVehicle(edata, econst);
	}

	// Transfer the data to the device.
	SyncDevice();

	// Compute the initial KNN for this group with itself.
	// Update the KNN database for the group.
	KNNBinningUpdateDBCUDA kernel( this, g_pWorld->GetBinData() );
	kernel.init();
	kernel.run();
	kernel.close();
	findKNearestNeighbors( this, m_pKNNSelf, g_pWorld->GetBinData(), this );
}

// ----------------------------------------------------------------------------
void CtfEnemyGroup::randomizeStartingPositionAndHeading(VehicleData &vehicleData, VehicleConst &vehicleConst)
{
    // randomize position on a ring between inner and outer radii
    // centered around the home base
    const float rRadius = frandom2 (gMinStartRadius, gMaxStartRadius);
    const float3 randomOnRing = float3_scalar_multiply(float3_RandomUnitVectorOnXZPlane (), rRadius);
	vehicleData.position = float3_add(gHomeBaseCenter, randomOnRing);

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
		randomizeHeadingOnXZPlane(vehicleData);
    //}
}

void CtfEnemyGroup::draw(void)
{
#if defined NO_DRAW
	return;
#endif

	// Draw all of the enemies
	float3 bodyColor = make_float3(0.6f, 0.4f, 0.4f); // redish

	VehicleConst vc;
	VehicleData vd;

	AgentGroupConst & vgc = gEnemies->GetAgentGroupConst();
	AgentGroupData & vgd = gEnemies->GetAgentGroupData();

#if defined ANNOTATION_LINES || defined ANNOTATION_TEXT
	// Temporary storage used for annotation.
	uint KNNIndices[g_knn];
	float KNNDistances[g_knn];
#endif

	// For each enemy...
	for( size_t i = 0; i < gEnemies->Size(); i++ )
	//for( size_t i = 0; i < gEnemies->Size(); i += 100 )
	{
		// Get its varialbe and constant data.
		vgc.getVehicleData( i, vc );
		vgd.getVehicleData( i, vd );

		// Draw the agent.
		drawBasic2dCircularVehicle( vc.radius, vd.position, vd.forward, vd.side, bodyColor );

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
			if( KNNIndices[j] < gEnemies->Size() )
			{
				vgd.getVehicleData( KNNIndices[j], vdOther );
				drawLine( vd.position, vdOther.position, make_float3( 1.f, 1.f, 1.f ) );
			}
		}
#endif
	}
}

void CtfEnemyGroup::update(const float currentTime, const float elapsedTime)
{
	//CUDAGroupSteerLibrary.steerToAvoidObstacles(*this, 3.0f, *gObstacles);

	// TODO: add time prediction to the seek kernel instead of the hack second parameter.
	// This should be accomplished with a pursuit kernel instead.
	//CUDAGroupSteerLibrary.steerForSeek(*this, gSeeker->position());

	//SyncDevice();
	// Force the host to pull data on next call to SyncHost().
	//SetSyncHost();	// <-- TODO: This should be replaced by appropriate setSynsHost calls in the kernel close() methods.

	// Update the positions in the KNNDatabase for the group.
	KNNBinningUpdateDBCUDA kernel( this, g_pWorld->GetBinData() );
	kernel.init();
	kernel.run();
	kernel.close();

	// Update the KNNDatabases
	findKNearestNeighbors( this, m_pKNNObstacles, g_pWorld->GetBinData(), gObstacles );
	findKNearestNeighbors( this, m_pKNNSelf, g_pWorld->GetBinData(), this );

	// Avoid collision with obstacles.
	steerToAvoidObstacles( this, gObstacles, m_pKNNObstacles, g_fMinTimeToObstacle, g_fWeightObstacleAvoidance, 0 );

	// Avoid collision with self.
	steerToAvoidNeighbors( this, m_pKNNSelf, this,  g_fMinTimeToCollision, g_fMinSeparationDistance, g_fWeightAvoidNeighbors, KERNEL_AVOID_OBSTACLES_BIT | KERNEL_AVOID_WALLS_BIT );

	// Pursue target.
	steerForPursuit( this, gSeeker->getVehicleData(), g_fMaxPursuitPredictionTime, g_fWeightPursuit, KERNEL_AVOID_OBSTACLES_BIT | KERNEL_AVOID_WALLS_BIT );

	// Maintain separation.
	steerForSeparation( this, m_pKNNSelf, this, g_fWeightSeparation, KERNEL_AVOID_OBSTACLES_BIT | KERNEL_AVOID_WALLS_BIT );

	// Apply steering.
	updateGroup( this, elapsedTime );

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
	_const.id = serialNumber;

    SimpleVehicle::reset ();  // reset the vehicle 

    setSpeed (3);             // speed along Forward direction.
    setMaxForce (3.0);        // steering force is clipped to this magnitude
    setMaxSpeed (3.0);        // velocity is clipped to this magnitude

    avoiding = false;         // not actively avoiding

	randomizeStartingPositionAndHeading( _data.position, _const.radius, _data.up, _data.forward, _data.side );  // new starting position

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
void randomizeStartingPositionAndHeading( float3 & position, float const radius, float3 & up, float3 & forward, float3 & side )
{
    // randomize position on a ring between inner and outer radii
    // centered around the home base
    const float rRadius = frandom2 ( gMinStartRadius, gMaxStartRadius );
    const float3 randomOnRing = float3_scalar_multiply( float3_RandomUnitVectorOnXZPlane(), rRadius );
    position =  float3_add( gHomeBaseCenter, randomOnRing );

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
        randomizeHeadingOnXZPlane( up, forward, side );
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

    const float3 goalOffset = float3_subtract(gHomeBaseCenter, position());
    const float goalDistance = float3_length(goalOffset);
    const float3 goalDirection = float3_scalar_divide(goalOffset, goalDistance);

    const bool goalIsAside = isAside (*this, gHomeBaseCenter, 0.5);

    // for annotation: loop over all and save result, instead of early return 
    bool xxxReturn = true;

	VehicleConst econst;
	VehicleData edata;

	AgentGroupConst & vgc = gEnemies->GetAgentGroupConst();
	AgentGroupData & vgd = gEnemies->GetAgentGroupData();

	// For each enemy...
	for( size_t i = 0; i < gEnemies->Size(); i++ )
	{
		// Get its data and const.
		vgc.getVehicleData( i, econst );
		vgd.getVehicleData( i, edata );

		const float eDistance = float3_distance ( position(), edata.position );

		/*
		// Check if we were tagged.
		if(eDistance < (econst.radius + radius()))
		{
			state = tagged;
			return true;
		}
		*/

		const float timeEstimate = 0.3f * eDistance / edata.speed;
		const float3 eFuture = edata.predictFuturePosition(timeEstimate);
		const float3 eOffset = float3_subtract(eFuture, position());
		const float alongCorridor = float3_dot(goalDirection, eOffset);
        const bool inCorridor = ((alongCorridor > -behindThreshold) && (alongCorridor < goalDistance));
        const float eForwardDistance = float3_dot(edata.forward, eOffset);

        // xxx temp move this up before the conditionals
		annotationXZCircle (econst.radius, eFuture, clearPathColor, 20); //xxx

        // consider as potential blocker if within the corridor
        if (inCorridor)
        {
            const float3 perp = float3_subtract(eOffset, float3_scalar_multiply(goalDirection, alongCorridor));
            const float acrossCorridor = float3_length(perp);
            if (acrossCorridor < sideThreshold)
            {
                // not a blocker if behind us and we are perp to corridor
				const float eFront = eForwardDistance + econst.radius;

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
					annotationLine (position(), edata.position, clearPathColor);
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
	const float3 behindBack = float3_scalar_multiply(forward(), -behindThreshold);
    const float3 pbb = float3_add(position(), behindBack);
    const float3 gun = localRotateForwardToSide (goalDirection);
	const float3 gn = float3_scalar_multiply(gun, sideThreshold);
    const float3 hbc = gHomeBaseCenter;
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
	const float3 boxFront = float3_scalar_multiply(forward(), minDistanceToCollision);
	const float3 FR = float3_subtract(float3_add(position(), boxFront), boxSide);
    const float3 FL = float3_add(float3_add(position(), boxFront), boxSide);
    const float3 BR = float3_subtract(position(), boxSide);
    const float3 BL = float3_add(position(), boxSide);
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

    const float goalDistance = float3_distance(gHomeBaseCenter, position());

	VehicleConst econst;
	VehicleData edata;

	AgentGroupConst & vgc = gEnemies->GetAgentGroupConst();
	AgentGroupData & vgd = gEnemies->GetAgentGroupData();

	// For each enemy...
	for( size_t i = 0; i < gEnemies->Size(); i++ )
	{
		// Get its data and const.
		vgc.getVehicleData( i, econst );
		vgd.getVehicleData( i, edata );

        const float3 eOffset = float3_subtract(edata.position, position());
        const float eDistance = float3_length(eOffset);

        const float eForwardDistance = float3_dot(forward(), eOffset);
        const float behindThreshold = radius() * 2;
        const bool behind = eForwardDistance < behindThreshold;
        if ((!behind) && (eDistance < 20))
        {
            if (eDistance < (goalDistance * 1.2)) //xxx
            {
				//float lookAheadTime = float3_length(toTarget) / (SPEED(offset) + target->speed);
                //const float timeEstimate = 0.5f * eDistance / edata.speed;//xxx
				const float timeEstimate = eDistance / (edata.speed + maxSpeed());//xxx
                //const float timeEstimate = 0.15f * eDistance / e.speed();//xxx
                const float3 future = edata.predictFuturePosition (timeEstimate);

                annotationXZCircle (econst.radius, future, evadeColor, 20); // xxx

                const float3 offset = float3_subtract(future, position());
                const float3 lateral = float3_perpendicularComponent(offset, forward());
                const float d = float3_length(lateral);
                const float weight = -1000 / (d * d);
				evade = float3_add(evade, float3_scalar_multiply(float3_scalar_divide(lateral, d), weight));
            }
        }
    }
    return evade;
}

float3 CtfSeeker::XXXsteerToEvadeAllDefenders (void)
{
    // sum up weighted evasion
    float3 evade = float3_zero();

	VehicleConst econst;
	VehicleData edata;

	AgentGroupConst & vgc = gEnemies->GetAgentGroupConst();
	AgentGroupData & vgd = gEnemies->GetAgentGroupData();

	// For each enemy...
	for( size_t i = 0; i < gEnemies->Size(); i++ )
	{
		// Get its data and const.
		vgc.getVehicleData( i, econst );
		vgd.getVehicleData( i, edata );

        const float3 eOffset = float3_subtract(edata.position, position());
        const float eDistance = float3_length(eOffset);

        // xxx maybe this should take into account e's heading? xxx // TODO: just that :)
        const float timeEstimate = 0.5f * eDistance / edata.speed; //xxx
        const float3 eFuture = edata.predictFuturePosition(timeEstimate);

        // annotation
        annotationXZCircle (econst.radius, eFuture, evadeColor, 20);

        // steering to flee from eFuture (enemy's future position)
        const float3 flee = xxxsteerForFlee (*this, eFuture);

        const float eForwardDistance = float3_dot(forward(), eOffset);
        const float behindThreshold = radius() * -2;

        const float distanceWeight = 4 / eDistance;
        const float forwardWeight = ((eForwardDistance > behindThreshold) ? 1.0f : 0.5f);

		const float3 adjustedFlee = float3_scalar_multiply(flee, distanceWeight * forwardWeight);

		evade = float3_add(evade, adjustedFlee);
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
	
	float3 const& seekerPosition = position();
	float const halfDim = 0.75f * gDim;

    // saved for annotation
    avoiding = !float3_equals(obstacleAvoidance, float3_zero());

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
		float3 seek = steerForSeek( *this, gHomeBaseCenter );
		seek.y = 0.f;
		wandering = false;
		return seek;

	}
	else
	{
		float3 wander = steerForWander( *this, elapsedTime );
		wander.y = 0.f;
		wandering = true;
		return wander;
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
        const float goalDistance = float3_distance(gHomeBaseCenter,position());
        const bool headingTowardGoal = isAhead (*this, gHomeBaseCenter, 0.98f);
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
        const float baseDistance = float3_distance(position(),gHomeBaseCenter);
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
    const float3 textOrigin = float3_add(position(), make_float3(0, 0.25, 0));
    std::ostringstream annote;
    annote << seekerStateString << std::endl;
    annote << std::setprecision(2) << std::setiosflags(std::ios::fixed)
           << speed() << std::ends;
    draw2dTextAt3dLocation (annote, textOrigin, gWhite);


    // display status in the upper left corner of the window
    std::ostringstream status;
    status << seekerStateString << std::endl;
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
    recordTrailVertex (currentTime, position());
}

// ----------------------------------------------------------------------------
// PlugIn for OpenSteerDemo


class CtfPlugIn : public PlugIn
{
private:

public:

    const char* name (void) {return "Capture the Flag";}

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
		gEnemies = new CtfEnemyGroup;
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
        const float3 goalOffset = float3_subtract(gHomeBaseCenter, OpenSteerDemo::camera.position());
        const float3 goalDirection = float3_normalize(goalOffset);
        const float3 cameraForward = OpenSteerDemo::camera.xxxls().forward();
        const float goalDot = float3_dot(cameraForward, goalDirection);
        const float blend = remapIntervalClip (goalDot, 1, 0, 0.5, 0);
        const float3 gridCenter = interpolate (blend,
                                             selected.position(),
                                             gHomeBaseCenter);
        OpenSteerDemo::gridUtility (gridCenter);

        // draw the seeker, obstacles and home base
        ctfSeeker->draw();
        //drawObstacles ();
		gObstacles->draw();
        drawHomeBase();

        // draw each enemy
        //for (int i = 0; i < ctfEnemyCount; i++) ctfEnemies[i]->draw ();
		gEnemies->draw();

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
		delete gEnemies;

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
        drawXZDisk (g_fHomeBaseRadius,    gHomeBaseCenter, baseColor, 20);
        drawXZDisk (g_fHomeBaseRadius/15, float3_add(gHomeBaseCenter, up), gBlack, 20);
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


