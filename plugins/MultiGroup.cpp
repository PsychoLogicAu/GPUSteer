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

#include "PluginCommon.h"

#include "OpenSteer/WallGroup.h"
#include "OpenSteer/CUDA/PolylinePathwayCUDA.cuh"

using namespace OpenSteer;

// ----------------------------------------------------------------------------
// forward declarations

class Group;

//class CtfObstacleGroup;

class MultiGroupWorld;
class MultiGroupSimulation;

// ----------------------------------------------------------------------------
// globals
MultiGroupSimulation *		g_pSimulation;
static uint					g_nNumGroups = 0;

// count the number of times the simulation has reset (e.g. for overnight runs)
static int					resetCount = 0;

class MultiGroupWorld : public SimulationWorld
{
private:
	// Bin data to be used for KNN lookups.
	KNNBinData *							m_pKNNBinData;
	//WallGroup *								m_pWallGroup;

public:
	MultiGroupWorld( SimulationParams * pSimulationParams, WorldParams * pWorldParams )
	:	SimulationWorld( pSimulationParams, pWorldParams ),
		m_pKNNBinData( NULL )/*,
		m_pWallGroup( NULL )*/
	{
		m_pKNNBinData = new KNNBinData( m_pWorldParams->m_u3Cells, m_pWorldParams->m_f3Dimensions, pSimulationParams->m_nSearchRadius );
		//m_pWallGroup = new WallGroup( m_pWorldParams->m_u3Cells, m_pSimulationParams->m_nKNW );

		//// Load the walls from a file.
		//m_pWallGroup->LoadFromFile( m_pWorldParams->m_strMapFilename.c_str() );
		//// Clip the walls to the edges of the cells.
		//m_pWallGroup->SplitWalls( m_pKNNBinData->hvCells() );

		// Send the data to the device.
		//m_pWallGroup->SyncDevice();

		// Update the KNN Database for the WallGroup.
		//updateKNNDatabase( m_pWallGroup, m_pKNNBinData );
	}

	~MultiGroupWorld( void )
	{
		SAFE_DELETE( m_pKNNBinData ); 
		//SAFE_DELETE( m_pWallGroup );
	}

	//WallGroup * GetWalls( void )	{ return m_pWallGroup; }

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
		//std::vector< float4 > const& start		= m_pWallGroup->GetWallGroupData().hvLineStart();
		//std::vector< float4 > const& mid		= m_pWallGroup->GetWallGroupData().hvLineMid();
		//std::vector< float4 > const& end		= m_pWallGroup->GetWallGroupData().hvLineEnd();
		//std::vector< float4 > const& normal		= m_pWallGroup->GetWallGroupData().hvLineNormal();

		//float3 const lineColor = { 1.f, 0.f, 0.f };

		//// For each line in the host data...
		//for( uint i = 0; i < start.size(); i++ )
		//{
		//	// Draw the line.
		//	drawLine( make_float3(start[i]), make_float3(end[i]), lineColor );
		//	// Draw the normal.
		//	drawLine( make_float3(mid[i]), float3_add( make_float3(mid[i]), make_float3(normal[i]) ), lineColor );
		//}
	}

	KNNBinData * GetBinData( void )		{ return m_pKNNBinData; }
};

class Group : public SimulationGroup
{
private:
	uint					m_iThisGroup;
	KNNData **				m_pKNNGroups;

	PolylinePathwayCUDA *	m_pPath;

	float3					m_f3BodyColor;

public:
	Group( SimulationParams * pSimulationParams, GroupParams * pGroupParams, uint const iThisGroup )
	:	SimulationGroup( pSimulationParams, pGroupParams ),
		m_pKNNGroups( NULL ),
		m_pPath( NULL ),
		m_f3BodyColor( pGroupParams->m_f3BodyColor ),
		m_iThisGroup( iThisGroup )
	{
		// Create the KNNData objects.
		m_pKNNGroups = new KNNData*[ g_nNumGroups ];
		for( uint i = 0; i < g_nNumGroups; i++ )
		{
			m_pKNNGroups[ i ] = new KNNData( m_pGroupParams->m_nNumAgents, m_pSimulationParams->m_nKNN );
		}

		// Create the path.
		if( ! m_pGroupParams->m_vecPathPoints.empty() )
		{
			m_pPath = new PolylinePathwayCUDA( m_pGroupParams->m_vecPathPoints, m_pGroupParams->m_fPathRadius, m_pGroupParams->m_bPathIsCyclic );
		}

		reset();
	}

	virtual ~Group(void)
	{
		for( size_t i = 0; i < g_nNumGroups; i++ )
		{
			SAFE_DELETE( m_pKNNGroups[ i ] );
		}
		SAFE_DELETE_ARRAY( m_pKNNGroups );
		
		SAFE_DELETE( m_pPath );
	}

	void reset(void);
	void draw(void);

	void preUpdate( void );
	void update(const float currentTime, const float elapsedTime);
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

class MultiGroupSimulation : public Simulation
{
	friend class Group;
	friend class MultiGroupPlugIn;

protected:
	MultiGroupWorld *		m_pWorld;
	Group **				m_pGroups;
	//CtfObstacleGroup *	m_pObstacles;

	CameraProxy *			m_pCameraProxy;

public:
	MultiGroupSimulation( void )
	:	m_pWorld( NULL ),
		m_pGroups( NULL ),
		m_pCameraProxy( NULL )
	{
		m_pCameraProxy = new CameraProxy;
	}

	virtual ~MultiGroupSimulation( void )
	{
		SAFE_DELETE( m_pWorld );
		SAFE_DELETE( m_pCameraProxy );

		for( uint i = 0; i < g_nNumGroups; i++ )
			SAFE_DELETE( m_pGroups[ i ] );
		SAFE_DELETE_ARRAY( m_pGroups );

	}

	virtual void load( char const* szFilename )
	{
		SAFE_DELETE( m_pWorld );
		for( uint i = 0; i < g_nNumGroups; i++ )
			SAFE_DELETE( m_pGroups[ i ] );
		SAFE_DELETE_ARRAY( m_pGroups );

		Simulation::load( szFilename );

		g_nNumGroups = m_SimulationParams.m_vecGroupParams.size();

		srand( m_SimulationParams.m_nSeed );
		OpenSteerDemo::maxSelectedVehicleIndex = m_SimulationParams.m_vecGroupParams[0].m_nNumAgents;

		// Create the groups.
		m_pGroups = new Group*[ g_nNumGroups ];
		for( uint i = 0; i < g_nNumGroups; i++ )
			m_pGroups[ i ] = new Group( &m_SimulationParams, &m_SimulationParams.m_vecGroupParams[ i ], i );

		// Create the world.
		m_pWorld = new MultiGroupWorld( &m_SimulationParams, &m_SimulationParams.m_WorldParams );
	}

	void update( float const currentTime, float const elapsedTime )
	{
		// Pre-update all the groups.
		for( uint i = 0; i < g_nNumGroups; i++ )
			m_pGroups[ i ]->preUpdate();

		// Update the groups
		for( uint i = 0; i < g_nNumGroups; i++ )
			m_pGroups[ i ]->update( currentTime, elapsedTime );

		// Update the camera proxy object.
		m_pCameraProxy->update( OpenSteerDemo::selectedVehicleIndex, m_pGroups[0] );
	}

	void draw( void )
	{
        // draw the groups
		for( uint i = 0; i < g_nNumGroups; i++ )
			m_pGroups[ i ]->draw();

		// draw the world
		m_pWorld->draw();

		// Draw the obstacles
		//m_pObstacles->draw();

		// display status in the upper left corner of the window
		std::ostringstream status;
		//status << std::left << std::setw( 25 ) << "No. obstacles: " << std::setw( 10 ) << g_pObstacles->Size() << std::endl;
		for( uint i = 0; i < g_nNumGroups; i++ )
		{
			status << std::left << std::setw( 25 ) << "No. agents group " << i << ": " << m_pGroups[i]->Size() << std::endl;
		}
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

void Group::reset(void)
{
	Clear();

	//static unsigned int id = 0;
	// Add the required number of enemies.
	while( m_nCount < m_pGroupParams->m_nNumAgents )
	{
		PluginBase agent;
		AgentData &aData = agent.getVehicleData();

		aData.speed = m_pGroupParams->m_fMaxSpeed;
		aData.maxForce = m_pGroupParams->m_fMaxForce;
		aData.maxSpeed = m_pGroupParams->m_fMaxSpeed;
		randomizeStartingPositionAndHeading2D( aData.position, aData.up, aData.direction, aData.side, m_pGroupParams->m_fMinStartRadius, m_pGroupParams->m_fMaxStartRadius, m_pGroupParams->m_f3StartPosition );
		
		bool success = AddAgent( aData );
	}

	// Transfer the data to the device.
	SyncDevice();
}

void Group::draw(void)
{
	if( g_bNoDraw )
	{
		return;
	}

	AgentData ad;

	// For each enemy...
	for( size_t i = 0; i < m_nCount; i++ )
	{
		// Get its varialbe and constant data.
		m_agentGroupData.getAgentData( i, ad );

		if( g_bNoDrawOutsideRange )
		{
			if( float3_distanceSquared( make_float3( ad.position ), make_float3( OpenSteerDemo::camera.position() ) ) > 18000.f )
				continue;
		}

		// Draw the agent.
		drawBasic2dCircularVehicle( ad.radius, make_float3(ad.position), make_float3(ad.direction), ad.side, m_f3BodyColor );

		//if( g_bDrawAnnotationWallLines )
		//{
		//	// Temporary storage for the KNN data.
		//	uint *	KNWIndices		= new uint[ m_pSimulationParams->m_nKNW ];
		//	float *	KNWDistances	= new float[ m_pSimulationParams->m_nKNW ];

		//	m_pKNNWalls->getAgentData( i, KNWIndices, KNWDistances );

		//	WallGroupData const& wgd = g_pSimulation->m_pWorld->GetWalls()->GetWallGroupData();

		//	std::vector< float4 > const& hvLineMid = wgd.hvLineMid();

		//	// Draw the KNW links.
		//	for( uint j = 0; j < m_pSimulationParams->m_nKNW; j++ )
		//	{
		//		if( KNWIndices[j] < wgd.size() )
		//		{
		//			float3 const& lineMid = make_float3(hvLineMid[KNWIndices[j]]);
		//			drawLine( make_float3(ad.position), lineMid, make_float3( 1.f, 1.f, 1.f ) );
		//		}
		//	}

		//	delete [] KNWIndices;
		//	delete [] KNWDistances;
		//}

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
			for( uint g = 0; g < g_nNumGroups; g++ )
			{
				// Temporary storage used for annotation.
				uint *	KNNIndices		= new uint[ m_pSimulationParams->m_nKNN ];
				float *	KNNDistances	= new float[ m_pSimulationParams->m_nKNN ];

				// Pull the KNN data for this agent from the nearest neighbor database.
				m_pKNNGroups[ g ]->getAgentData( i, KNNIndices, KNNDistances );

				AgentData adOther;

				// Draw the KNN links.
				for( uint j = 0; j < m_pSimulationParams->m_nKNN; j++ )
				{
					if( KNNIndices[j] < g_pSimulation->m_pGroups[ g ]->Size() )
					{
						g_pSimulation->m_pGroups[ g ]->GetAgentGroupData().getAgentData( KNNIndices[j], adOther );
						drawLine( make_float3( ad.position ), make_float3( adOther.position ), make_float3( 1.f, 1.f, 1.f ) );
					}
				}

				delete [] KNNIndices;
				delete [] KNNDistances;
			}
		}
	}
}

void Group::preUpdate( void )
{
	// Wrap the world.
	wrapWorld( this, m_pSimulationParams->m_WorldParams.m_f3Dimensions );

	// Update the positions in the KNNDatabase for the group.
	updateKNNDatabase( this, g_pSimulation->m_pWorld->GetBinData() );

	// Reset the applied kernels.
	CUDA_SAFE_CALL( cudaMemset( m_agentGroupData.pdAppliedKernels(), 0, m_nCount * sizeof(uint) ) );
}

void Group::update( const float currentTime, const float elapsedTime )
{
	// Update the KNNDatabases for the groups.
	for( uint i = 0; i < g_nNumGroups; i++ )
	{
		findKNearestNeighbors( this, m_pKNNGroups[ i ], g_pSimulation->m_pWorld->GetBinData(), g_pSimulation->m_pGroups[ i ], m_pSimulationParams->m_nSearchRadius );
	}

	// Avoid collisions with walls.
	//steerToAvoidWalls( this, m_pKNNWalls, g_pSimulation->m_pWorld->GetWalls(), m_pSimulationParams->m_fMinTimeToWall, m_pSimulationParams->m_fWeightAvoidWalls, m_pSimulationParams->m_nMaskAvoidWalls );

	// Avoid collision with obstacles.
	//steerToAvoidObstacles( this, gObstacles, m_pKNNObstacles, g_fMinTimeToObstacle, g_fWeightObstacleAvoidance, g_maskObstacleAvoidance );

	// Avoid collision with groups.
	for( uint i = 0; i < g_nNumGroups; i++ )
	{
		if( i == m_iThisGroup )
			continue;

		steerToAvoidNeighbors( this, m_pKNNGroups[i], g_pSimulation->m_pGroups[i], m_pSimulationParams->m_fMinTimeToCollision, m_pSimulationParams->m_fMinSeparationDistance, m_pSimulationParams->m_bAvoidCloseNeighbors, m_pSimulationParams->m_fWeightAvoidNeighbors, m_pSimulationParams->m_nMaskAvoidNeighbors );
	}

	// Pursue target.
	//steerForPursuit( this, gSeeker->getVehicleData(), g_fMaxPursuitPredictionTime, g_fWeightPursuit, g_maskPursuit );

	// Flocking.
	for( uint i = 0; i < g_nNumGroups; i++ )
	{
		// Maintain separation from all groups.
		steerForSeparation( this, m_pKNNGroups[i], g_pSimulation->m_pGroups[i], m_pSimulationParams->m_fMinSeparationDistance, m_pSimulationParams->m_fMaxSeparationDistance, m_pSimulationParams->m_fCosMaxFlockingAngle, m_pSimulationParams->m_fWeightSeparation, m_pSimulationParams->m_nMaskSeparation );
	}
	// Maintain alignment and cohesion only with self.
	steerForAlignment( this, m_pKNNGroups[m_iThisGroup], this, m_pSimulationParams->m_fMinFlockingDistance, m_pSimulationParams->m_fMaxFlockingDistance, m_pSimulationParams->m_fCosMaxFlockingAngle, m_pSimulationParams->m_fWeightAlignment, m_pSimulationParams->m_nMaskAlignment );
	steerForCohesion( this, m_pKNNGroups[m_iThisGroup], this, m_pSimulationParams->m_fMinFlockingDistance, m_pSimulationParams->m_fMaxFlockingDistance, m_pSimulationParams->m_fCosMaxFlockingAngle, m_pSimulationParams->m_fWeightCohesion, m_pSimulationParams->m_nMaskCohesion );

	if( m_pPath )
	{
		steerToFollowPath( this, m_pPath, m_pSimulationParams->m_fPathPredictionTime, m_pSimulationParams->m_fWeightFollowPath, m_pSimulationParams->m_nMaskFollowPath );
		steerForSeek( this, m_pPath->hvPoints()[1], m_pSimulationParams->m_fWeightSeek, m_pSimulationParams->m_nMaskSeek );
	}

	// Apply steering.
	updateGroup( this, elapsedTime );
	//updateGroupWithAntiPenetration( this, m_pKNNWalls, g_pSimulation->m_pWorld->GetWalls(), elapsedTime );

	// Force anti-penetration.
	for( uint i = 0; i < g_nNumGroups; i++ )
	{
		antiPenetrationAgents( this, m_pKNNGroups[i], g_pSimulation->m_pGroups[i], m_pSimulationParams->m_nMaskAntiPenetrationAgents );
	}
}

// ----------------------------------------------------------------------------
// PlugIn for OpenSteerDemo


class MultiGroupPlugIn : public PlugIn
{
private:

public:

    const char* name (void) {return "Multi Group Right Angle";}

    float selectionOrderSortKey (void) {return 0.01f;}

    virtual ~MultiGroupPlugIn() {} // be more "nice" to avoid a compiler warning

    void open (void)
    {
		OpenSteerDemo::setAnnotationOff();

		int numDevices;
		cudaGetDeviceCount(&numDevices);

		// TODO: more intelligent selection of the CUDA device.
		CUDA_SAFE_CALL( cudaSetDevice( 0 ) );

		g_pSimulation = new MultiGroupSimulation;

        // initialize camera
        OpenSteerDemo::init2dCamera( *g_pSimulation->m_pCameraProxy );
		OpenSteerDemo::camera.mode = Camera::cmFixed;
        OpenSteerDemo::camera.fixedTarget = make_float3(1, 0, 0);
        OpenSteerDemo::camera.fixedPosition = make_float3(0, 100, 0);

		reset();
		resetCount = 0;

		all.push_back ( g_pSimulation->m_pCameraProxy );
    }

	void reset (void)
    {
        // count resets
        resetCount++;

		g_pSimulation->reset();
		g_pSimulation->load( "MultiGroup1VStationary.params" );

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
    std::vector<PluginBase*> all;
};


MultiGroupPlugIn gMultiGroupPlugIn;


// ----------------------------------------------------------------------------


