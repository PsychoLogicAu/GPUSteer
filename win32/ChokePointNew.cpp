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
#include "OpenSteer/Simulation.h"
#include "OpenSteer/SimpleVehicle.h"
#include "OpenSteer/OpenSteerDemo.h"
#include "OpenSteer/Proximity.h"
//#include "OpenSteer/ObstacleGroup.h"

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

#ifndef M_PI
	#define M_PI       3.14159265358979323846
#endif

//#define ANNOTATION_LINES
//#define ANNOTATION_WALL_LINES
//#define ANNOTATION_TEXT
//#define ANNOTATION_CELLS
//#define NO_DRAW
//#define NO_DRAW_OBSTACLES
//#define NO_DRAW_OUTSIDE_RANGE

static bool	g_bDrawAnnotationLines		= false;
static bool	g_bDrawAnnotationWallLines	= false;
static bool	g_bDrawAnnotationText		= false;
static bool	g_bDrawAnnotationCells		= false;
static bool	g_bNoDraw					= false;
static bool	g_bNoDrawOutsideRange		= true;
static bool	g_bNoDrawObstacles			= false;

// ----------------------------------------------------------------------------
// forward declarations
class CtfBase;
class CtfProxy;
class CtfGroup;

//class CtfObstacleGroup;

class CtfWorld;
class CtfSimulation;

// ----------------------------------------------------------------------------
// globals
CtfSimulation *		g_pSimulation;

// count the number of times the simulation has reset (e.g. for overnight runs)
int resetCount = 0;

// Function prototypes.
void randomizeStartingPositionAndHeadingCtf( float4 & position, float3 & up, float4 & forward, float3 & side, float const minRadius, float const maxRadius, float3 const& startPosition );

void randomizeStartingPositionAndHeadingCtf( float4 & position, float3 & up, float4 & forward, float3 & side, float const minRadius, float const maxRadius, float3 const& startPosition )
{
    // randomize position on a ring between inner and outer radii
    // centered around the home base
    const float rRadius = frandom2 ( minRadius, maxRadius );
	float3 const randomOnRing = float3_scalar_multiply( float3_RandomUnitVectorOnXZPlane(), rRadius );

    position = make_float4( float3_add( startPosition, randomOnRing ), 0.f );

	float3 newForward;
    randomizeHeading( up, newForward, side );
	up = make_float3( 0.f, 1.f, 0.f );
	newForward.y = 0.f;
	forward = make_float4( newForward, 0.f );
}

#pragma region obsolete
/*
// Using cell diameter of 7

//const int	gEnemyCount					= 100;
//const float gDim						= 50;
//const int	gCells						= 10;

//const int gEnemyCount					= 1000;
//const float gDim						= 200;
//const int gCells						= 50;

//const int gEnemyCount					= 10000;
////const float gDim						= 635;
//const float gDim						= 935;
////const int gCells						= 91;
//const int gCells						= 200;

const int gEnemyCount					= 100000;
const float gDim						= 2000;
//const int gCells						= 285;
//const int gCells						= 512;
const int gCells						= 2048;

//const int gEnemyCount					= 500000;
//const float gDim						= 3000;
////const int gCells						= 907;
////const int gCells						= 1814;
////const int gCells						= 1024;
//const int gCells						= 1536;

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
uint const	g_searchRadiusObstacles		= 3;		// Distance in cells to search for obstacles.
uint const	g_searchRadiusWalls			= 3;		// Distance in cells to search for obstacles.

float const g_fMaxPursuitPredictionTime	= 10.0f;		// Look-ahead time for pursuit.
float const g_fMinTimeToCollision		= 2.0f;		// Look-ahead time for neighbor avoidance.
float const g_fMinTimeToObstacle		= 5.0f;		// Look-ahead time for obstacle avoidance.
float const g_fMinTimeToWall			= 10.0f;		// Look-ahead time for wall avoidance.
float const g_fPathPredictionTime		= 5.f;		// Point on path in future to aim for.

float const g_fMinSeparationDistance	= 0.5f;		// Mini
float const g_fMaxSeparationDistance	= 1.2f;		// Maximum range for separation behavior.
float const g_fMinFlockingDistance		= 0.5f;
float const g_fMaxFlockingDistance		= 7.f;
float const g_fCosMaxFlockingAngle		= cosf(150 * (float)M_PI / 180.f);	// 350 degrees - used in "An efficient GPU implementation for large scale individual-based simulation of collective behavior"

// Weights for behaviors.
float const	g_fWeightAlignment			= 2.5f;
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
uint const	g_maskSeparation			= KERNEL_ANTI_PENETRATION_WALL;//0;//KERNEL_AVOID_WALLS_BIT;

uint const	g_maskSeek					= KERNEL_AVOID_WALLS_BIT;// | KERNEL_SEPARATION_BIT;
uint const	g_maskFlee					= KERNEL_AVOID_OBSTACLES_BIT | KERNEL_AVOID_WALLS_BIT | KERNEL_AVOID_NEIGHBORS_BIT;
uint const	g_maskPursuit				= KERNEL_AVOID_OBSTACLES_BIT | KERNEL_AVOID_WALLS_BIT | KERNEL_AVOID_NEIGHBORS_BIT;
uint const	g_maskEvade					= KERNEL_AVOID_OBSTACLES_BIT | KERNEL_AVOID_WALLS_BIT | KERNEL_AVOID_NEIGHBORS_BIT;

uint const	g_maskFollowPath			= KERNEL_SEPARATION_BIT;

uint const	g_maskObstacleAvoidance		= 0;
uint const	g_maskNeighborAvoidance		= 0;
uint const	g_maskWallAvoidance			= 0;

float const	g_fMaxSpeed					= 3.f;

// Do not shove an agent which has been pushed out from a wall.
uint const	g_maskAntiPenetrationAgents	= KERNEL_AVOID_WALLS_BIT | KERNEL_ANTI_PENETRATION_WALL;

const float3 gWorldSize					= make_float3( gDim, 10.f, gDim );
const uint3 gWorldCells					= make_uint3( gCells, 1, gCells );

// Start position.
float3 const g_f3StartBaseCenter		= make_float3( 0.f, 0.f, 0.25f * gWorldSize.z );
float const g_fMinStartRadius			= 0.0f;
float const g_fMaxStartRadius			= 0.20f * min( gWorldSize.x, gWorldSize.z );

// Goal position.
float3 const g_f3GoalPosition			= make_float3( 0.f, 0.f, -2.f * gWorldSize.z );

float3 const g_f3HomeBaseCenter			= make_float3( 0.f, 0.f, 0.f );
const float g_fHomeBaseRadius			= 1.5f;



//const float g_fMaxStartRadius				= 60.0f;

*/
#pragma endregion

class CtfWorld : public SimulationWorld
{
private:
	// Bin data to be used for KNN lookups.
	KNNBinData *							m_pKNNBinData;
	WallGroup *								m_pWallGroup;

public:
	CtfWorld( SimulationParams * pSimulationParams, WorldParams * pWorldParams )
	:	SimulationWorld( pSimulationParams, pWorldParams ),
		m_pKNNBinData( NULL ),
		m_pWallGroup( NULL )
	{
		m_pKNNBinData = new KNNBinData( m_pWorldParams->m_u3Cells, m_pWorldParams->m_f3Dimensions, pSimulationParams->m_nSearchRadius );
		m_pWallGroup = new WallGroup( m_pWorldParams->m_u3Cells, m_pSimulationParams->m_nKNW );

		// Load the walls from a file.
		m_pWallGroup->LoadFromFile( m_pWorldParams->m_strMapFilename.c_str() );
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
		if( g_bDrawAnnotationCells )
		{
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
		}

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

class CtfGroup : public SimulationGroup
{
private:
	KNNData *				m_pKNNSelf;
	//KNNData *				m_pKNNObstacles;
	KNNData *				m_pKNNWalls;

	PolylinePathwayCUDA *	m_pPath;

public:
	CtfGroup( SimulationParams * pSimulationParams, GroupParams * pGroupParams )
	:	SimulationGroup( pSimulationParams, pGroupParams ),
		m_pKNNSelf( NULL ),
		//m_pKNNObstacles( NULL ),
		m_pKNNWalls( NULL )
	{
		// Create the KNNData objects.
		m_pKNNSelf = new KNNData( m_pGroupParams->m_nNumAgents, m_pSimulationParams->m_nKNN );
		//m_pKNNObstacles = new KNNData( m_pGroupParams->m_nNumAgents, m_pSimulationParams->m_nKNO );
		m_pKNNWalls = new KNNData( m_pGroupParams->m_nNumAgents, m_pSimulationParams->m_nKNW );

		// Create the path.
		m_pPath = new PolylinePathwayCUDA( m_pGroupParams->m_vecPathPoints, m_pGroupParams->m_fPathRadius, m_pGroupParams->m_bPathIsCyclic );

		reset();
	}

	virtual ~CtfGroup(void)
	{
		SAFE_DELETE( m_pKNNSelf );
		//SAFE_DELETE( m_pKNNObstacles );
		SAFE_DELETE( m_pKNNWalls );
		SAFE_DELETE( m_pPath );
	}

	void reset(void);
	void draw(void);

	void update(const float currentTime, const float elapsedTime);
};

// ----------------------------------------------------------------------------
// This PlugIn uses two vehicle types: CtfSeeker and CtfEnemy.  They have a
// common base class: CtfBase which is a specialization of SimpleVehicle.
class CtfBase : public SimpleVehicle
{
public:
	// for draw method
    float3 bodyColor;

    // constructor
    CtfBase () {reset ();}

	// reset state
	void reset (void)
	{
		_data.id = serialNumber;

		SimpleVehicle::reset ();  // reset the vehicle 
	}

	// ----------------------------------------------------------------------------
	// draw this character/vehicle into the scene
	void draw (void)
	{
		drawBasic2dCircularVehicle (*this, bodyColor);
		//drawTrail ();
	}
};

class CtfProxy : public CtfBase
{
public:
    // constructor
    CtfProxy (){}

	// Pull the latest data from the selected agent.
    void update( uint const selectedAgent, CtfGroup * pGroup )
	{
		pGroup->GetAgentGroupData().getAgentData( selectedAgent, _data );
	}
};
/*
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
*/

class CtfSimulation : public Simulation
{
	friend class CtfGroup;
	friend class CtfPlugIn;

protected:
	CtfWorld *			m_pWorld;
	CtfGroup *			m_pGroup;
	//CtfObstacleGroup *	m_pObstacles;

	CtfProxy *			m_pCameraProxy;

public:
	CtfSimulation( void )
	:	m_pWorld( NULL ),
		m_pGroup( NULL ),
		//m_pObstacles( NULL ),
		m_pCameraProxy( NULL )
	{
		m_pCameraProxy = new CtfProxy;
	}

	virtual ~CtfSimulation( void )
	{
		SAFE_DELETE( m_pWorld );
		SAFE_DELETE( m_pGroup );
		//SAFE_DELETE( m_pObstacles );
		SAFE_DELETE( m_pCameraProxy );
	}

	virtual void load( char const* szFilename )
	{
		Simulation::load( szFilename );

		srand( m_SimulationParams.m_nSeed );
		OpenSteerDemo::maxSelectedVehicleIndex = m_SimulationParams.m_vecGroupParams[0].m_nNumAgents;

		SAFE_DELETE( m_pWorld );
		SAFE_DELETE( m_pGroup );

		m_pWorld = new CtfWorld( &m_SimulationParams, &m_SimulationParams.m_WorldParams );
		m_pGroup = new CtfGroup( &m_SimulationParams, &m_SimulationParams.m_vecGroupParams[0] );
	}

	void update( float const currentTime, float const elapsedTime )
	{
		// Update the boids group
		m_pGroup->update(currentTime, elapsedTime);

		// Update the camera proxy object.
		m_pCameraProxy->update( OpenSteerDemo::selectedVehicleIndex, m_pGroup );
	}

	void draw( void )
	{
        // draw the enemy
		m_pGroup->draw();

		// draw the world
		m_pWorld->draw();

		// Draw the obstacles
		//m_pObstacles->draw();

		// display status in the upper left corner of the window
		std::ostringstream status;
		//status << std::left << std::setw( 25 ) << "No. obstacles: " << std::setw( 10 ) << g_pObstacles->Size() << std::endl;
		status << std::left << std::setw( 25 ) << "No. agents: " << m_pGroup->Size() << std::endl;
		status << std::left << std::setw( 25 ) << "World dim: " << m_SimulationParams.m_WorldParams.m_f3Dimensions << std::endl;
		status << std::left << std::setw( 25 ) << "World cells: " << m_SimulationParams.m_WorldParams.m_u3Cells << std::endl;
		status << std::left << std::setw( 25 ) << "Search radius: " << m_SimulationParams.m_nSearchRadius << std::endl;
		status << std::left << std::setw( 25 ) << "Camera proxy position: " << make_float3( m_pCameraProxy->position() ) << std::endl;
		status << std::left << std::setw( 25 ) << "Reset count: " << resetCount << std::endl;
		status << std::ends;
		const float h = drawGetWindowHeight ();
		const float3 screenLocation = make_float3(10, h-50, 0);
		draw2dTextAt2dLocation (status, screenLocation, gGray80);
	}
};

#define testOneObstacleOverlap(radius, center)					\
{																\
    float d = float3_distance (od.position, center);			\
    float clearance = d - (od.radius + (radius));				\
    if (minClearance > clearance) minClearance = clearance;		\
}

void CtfGroup::reset(void)
{
	Clear();

	//static unsigned int id = 0;
	// Add the required number of enemies.
	while( Size() < m_pGroupParams->m_nNumAgents )
	{
		CtfBase agent;
		AgentData &aData = agent.getVehicleData();

		aData.speed = m_pGroupParams->m_fMaxSpeed;
		aData.maxForce = m_pGroupParams->m_fMaxForce;
		aData.maxSpeed = m_pGroupParams->m_fMaxSpeed;
		randomizeStartingPositionAndHeadingCtf( aData.position, aData.up, aData.direction, aData.side, m_pGroupParams->m_fMinStartRadius, m_pGroupParams->m_fMaxStartRadius, m_pGroupParams->m_f3StartPosition );
		
		bool success = AddAgent( aData );
	}

	// Transfer the data to the device.
	SyncDevice();

	// Compute the initial KNN for this group with itself.
	// Update the KNN database for the group.
	updateKNNDatabase( this, g_pSimulation->m_pWorld->GetBinData() );
	findKNearestNeighbors( this, m_pKNNSelf, g_pSimulation->m_pWorld->GetBinData(), this, m_pSimulationParams->m_nSearchRadius );
}

void CtfGroup::draw(void)
{
	if( g_bNoDraw )
	{
		return;
	}

	// Draw all of the enemies
	float3 bodyColor = make_float3(0.6f, 0.4f, 0.4f); // redish

	AgentData ad;

	// For each enemy...
	for( size_t i = 0; i < m_nCount; i++ )
	{
		// Get its varialbe and constant data.
		m_agentGroupData.getAgentData( i, ad );

		if( g_bNoDrawOutsideRange )
		{
			if( float3_distanceSquared( make_float3( ad.position ), make_float3( OpenSteerDemo::camera.position() ) ) > 20000.f )
				continue;
		}

		// Draw the agent.
		drawBasic2dCircularVehicle( ad.radius, make_float3(ad.position), make_float3(ad.direction), ad.side, bodyColor );

		if( g_bDrawAnnotationWallLines )
		{
			// Temporary storage for the KNN data.
			uint *	KNWIndices		= new uint[ m_pSimulationParams->m_nKNW ];
			float *	KNWDistances	= new float[ m_pSimulationParams->m_nKNW ];

			m_pKNNWalls->getAgentData( i, KNWIndices, KNWDistances );

			WallGroupData const& wgd = g_pSimulation->m_pWorld->GetWalls()->GetWallGroupData();

			std::vector< float4 > const& hvLineMid = wgd.hvLineMid();

			// Draw the KNW links.
			for( uint j = 0; j < m_pSimulationParams->m_nKNW; j++ )
			{
				if( KNWIndices[j] < wgd.size() )
				{
					float3 const& lineMid = make_float3(hvLineMid[KNWIndices[j]]);
					drawLine( make_float3(ad.position), lineMid, make_float3( 1.f, 1.f, 1.f ) );
				}
			}

			delete [] KNWIndices;
			delete [] KNWDistances;
		}

		if( g_bDrawAnnotationText )
		{
			// annotate the agent with useful data.
			const float3 textOrigin = float3_add( make_float3(ad.position), make_float3( 0, 0.25, 0 ) );
			std::ostringstream annote;

			// Write this agent's index.
			annote << i << std::endl;
			annote << std::ends;

			draw2dTextAt3dLocation (annote, textOrigin, gWhite);
		}

		if( g_bDrawAnnotationLines )
		{
			// Temporary storage used for annotation.
			uint *	KNNIndices		= new uint[ m_pSimulationParams->m_nKNN ];
			float *	KNNDistances	= new float[ m_pSimulationParams->m_nKNN ];

			// Pull the KNN data for this agent from the nearest neighbor database.
			m_pKNNSelf->getAgentData( i, KNNIndices, KNNDistances );

			AgentData adOther;

			// Draw the KNN links.
			for( uint j = 0; j < m_pSimulationParams->m_nKNN; j++ )
			{
				if( KNNIndices[j] < m_nCount )
				{
					m_agentGroupData.getAgentData( KNNIndices[j], adOther );
					drawLine( make_float3(ad.position), make_float3(adOther.position), make_float3( 1.f, 1.f, 1.f ) );
				}
			}

			delete [] KNNIndices;
			delete [] KNNDistances;
		}
	}
}

void CtfGroup::update(const float currentTime, const float elapsedTime)
{
	// Wrap the world.
	wrapWorld( this, m_pSimulationParams->m_WorldParams.m_f3Dimensions );

	// Update the positions in the KNNDatabase for the group.
	updateKNNDatabase( this, g_pSimulation->m_pWorld->GetBinData() );

	// Reset the applied kernels.
	CUDA_SAFE_CALL( cudaMemset( m_agentGroupData.pdAppliedKernels(), 0, m_nCount * sizeof(uint) ) );

	// Update the KNNDatabases
	//findKNearestNeighbors( this, m_pKNNObstacles, g_pSimulation->m_pWorld->GetBinData(), gObstacles, g_searchRadiusObstacles );
	findKNearestNeighbors( this, m_pKNNSelf, g_pSimulation->m_pWorld->GetBinData(), this, m_pSimulationParams->m_nSearchRadius );
	findKNearestNeighbors( this, m_pKNNWalls, g_pSimulation->m_pWorld->GetBinData(), g_pSimulation->m_pWorld->GetWalls(), m_pSimulationParams->m_nSearchRadius );

	// Avoid collisions with walls.
	steerToAvoidWalls( this, m_pKNNWalls, g_pSimulation->m_pWorld->GetWalls(), m_pSimulationParams->m_fMinTimeToWall, m_pSimulationParams->m_fWeightAvoidWalls, m_pSimulationParams->m_nMaskAvoidWalls );

	// Avoid collision with obstacles.
	//steerToAvoidObstacles( this, gObstacles, m_pKNNObstacles, g_fMinTimeToObstacle, g_fWeightObstacleAvoidance, g_maskObstacleAvoidance );

	// Avoid collision with self.
	//steerToAvoidNeighbors( this, m_pKNNSelf, this,  g_fMinTimeToCollision, g_fMinSeparationDistance, false, g_fWeightAvoidNeighbors, g_maskNeighborAvoidance );

	// Pursue target.
	//steerForPursuit( this, gSeeker->getVehicleData(), g_fMaxPursuitPredictionTime, g_fWeightPursuit, g_maskPursuit );

	// Flocking.
	steerForSeparation( this, m_pKNNSelf, this, m_pSimulationParams->m_fMinSeparationDistance, m_pSimulationParams->m_fMaxSeparationDistance, m_pSimulationParams->m_fCosMaxFlockingAngle, m_pSimulationParams->m_fWeightSeparation, KERNEL_ANTI_PENETRATION_WALL/*m_pSimulationParams->m_nMaskSeparation*/ );
	steerForAlignment( this, m_pKNNSelf, this, m_pSimulationParams->m_fMinFlockingDistance, m_pSimulationParams->m_fMaxFlockingDistance, m_pSimulationParams->m_fCosMaxFlockingAngle, m_pSimulationParams->m_fWeightAlignment, m_pSimulationParams->m_nMaskAlignment );
	steerForCohesion( this, m_pKNNSelf, this, m_pSimulationParams->m_fMinFlockingDistance, m_pSimulationParams->m_fMaxFlockingDistance, m_pSimulationParams->m_fCosMaxFlockingAngle, m_pSimulationParams->m_fWeightCohesion, m_pSimulationParams->m_nMaskCohesion );

	//float3 seekPoint = make_float3( 0.f, 0.f, 300.f );
	steerToFollowPath( this, m_pPath, m_pSimulationParams->m_fPathPredictionTime, m_pSimulationParams->m_fWeightFollowPath, m_pSimulationParams->m_nMaskFollowPath );
	steerForSeek( this, m_pPath->hvPoints()[1], m_pSimulationParams->m_fWeightSeek, m_pSimulationParams->m_nMaskSeek );

	// Apply steering.
//	updateGroup( this, elapsedTime );
	updateGroupWithAntiPenetration( this, m_pKNNWalls, g_pSimulation->m_pWorld->GetWalls(), elapsedTime );

	// Force anti-penetration.
	//antiPenetrationWall( this, m_pKNNWalls, g_pWorld->GetWalls(), elapsedTime, 0 );
	antiPenetrationAgents( this, m_pKNNSelf, this, m_pSimulationParams->m_nMaskAntiPenetrationAgents );
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

		int numDevices;
		cudaGetDeviceCount(&numDevices);

		// TODO: more intelligent selection of the CUDA device.
		CUDA_SAFE_CALL( cudaSetDevice( 0 ) );

		g_pSimulation = new CtfSimulation;

        // initialize camera
        OpenSteerDemo::init2dCamera( *g_pSimulation->m_pCameraProxy );
        OpenSteerDemo::camera.mode = Camera::cmFixedDistanceOffset;
        OpenSteerDemo::camera.fixedTarget = make_float3(15, 0, 0);
        OpenSteerDemo::camera.fixedPosition = make_float3(80, 60, 0);

		reset();
		resetCount = 0;

		all.push_back ( g_pSimulation->m_pCameraProxy );
    }

	void reset (void)
    {
        // count resets
        resetCount++;

		g_pSimulation->reset();
		g_pSimulation->load( "ChokePoint.params" );

        // make camera jump immediately to new position
        OpenSteerDemo::camera.doNotSmoothNextMove ();
    }

    void update (const float currentTime, const float elapsedTime)
    {
		// Update the simulation.
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

        // clear the group of all vehicles
        all.clear();
    }

    void handleFunctionKeys (int keyNumber)
    {
        switch (keyNumber)
        {
		case 1:
			g_bDrawAnnotationLines = ! g_bDrawAnnotationLines;
			break;
		case 2:
			g_bDrawAnnotationWallLines = ! g_bDrawAnnotationWallLines;
			break;
		case 3:
			g_bDrawAnnotationText = ! g_bDrawAnnotationText;
			break;
		case 4:
			g_bDrawAnnotationCells = ! g_bDrawAnnotationCells;
			break;
		case 5:
			g_bNoDraw = ! g_bNoDraw;
			break;
		case 6:
			g_bNoDrawObstacles = ! g_bNoDrawObstacles;
			break;
		case 7:
			g_bNoDrawOutsideRange = ! g_bNoDrawOutsideRange;
			break;
        }
    }

    void printMiniHelpForFunctionKeys (void)
    {
        std::ostringstream message;
        message << "Function keys handled by ";
        message << '"' << name() << '"' << ':' << std::ends;
        OpenSteerDemo::printMessage (message);
        OpenSteerDemo::printMessage ("  F1     Toggle draw annotation lines.");
        OpenSteerDemo::printMessage ("  F2     Toggle draw annotation wall lines.");
		OpenSteerDemo::printMessage ("  F3     Toggle draw annotation text.");
		OpenSteerDemo::printMessage ("  F4     Toggle draw annotation cells.");
		OpenSteerDemo::printMessage ("  F5     Toggle draw group.");
		OpenSteerDemo::printMessage ("  F6     Toggle draw obstacles.");
		OpenSteerDemo::printMessage ("  F7     Toggle draw outside range.");
        OpenSteerDemo::printMessage ("");
    }

    const AVGroup& allVehicles (void) {return (const AVGroup&) all;}

    // a group (STL vector) of all vehicles in the PlugIn
    std::vector<CtfBase*> all;
};


CtfPlugIn gCtfPlugIn;


// ----------------------------------------------------------------------------


