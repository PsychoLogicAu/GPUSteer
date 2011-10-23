#ifndef OPENSTEER_PLUGINCOMMON_H
#define OPENSTEER_PLUGINCOMMON_H

#include "OpenSteer/Globals.h"

#include "OpenSteer/OpenSteerDemo.h"
#include "OpenSteer/Simulation.h"
#include "OpenSteer/AgentData.h"
#include "OpenSteer/AgentGroup.h"
//#include "OpenSteer/ObstacleGroup.h"

#include "OpenSteer/SimpleVehicle.h"

#include "OpenSteer/CUDA/GroupSteerLibrary.cuh"

// Include the required KNN headers.
#include "OpenSteer/CUDA/KNNBinData.cuh"

namespace OpenSteer
{
static bool	g_bDrawAnnotationLines		= false;
static bool	g_bDrawAnnotationWallLines	= false;
static bool	g_bDrawAnnotationText		= false;
static bool	g_bDrawAnnotationCells		= false;
static bool	g_bNoDraw					= false;
static bool	g_bNoDrawOutsideRange		= true;
static bool	g_bNoDrawObstacles			= false;

class PluginBase;
class CameraProxy;

#ifndef min
	#define min(a,b)            (((a) < (b)) ? (a) : (b))
#endif

#ifndef M_PI
	#define M_PI       3.14159265358979323846
#endif




// Function prototypes.
void randomizeStartingPositionAndHeading2D( float4 & position, float3 & up, float4 & forward, float3 & side, float const minRadius, float const maxRadius, float3 const& startPosition );
void randomizeStartingPositionAndHeading3D( float4 & position, float3 & up, float4 & forward, float3 & side, float const minRadius, float const maxRadius, float3 const& startPosition );


// ----------------------------------------------------------------------------
// Common base class for simulations: PluginBase which is a specialization of SimpleVehicle.
class PluginBase : public SimpleVehicle
{
public:
	// for draw method
    float3 bodyColor;

    // constructor
    PluginBase () {reset ();}

	// reset state
	void reset (void)
	{
		_data.id = serialNumber;

		SimpleVehicle::reset ();  // reset the vehicle 

		clearTrailHistory ();     // prevent long streaks due to teleportation
	}

	// ----------------------------------------------------------------------------
	// draw this character/vehicle into the scene
	void draw (void)
	{
		drawBasic2dCircularVehicle (*this, bodyColor);
		//drawTrail ();
	}
};

class CameraProxy : public PluginBase
{
public:
    // constructor
    CameraProxy (){}

	// Pull the latest data from the selected agent.
    void update( uint const selectedAgent, AgentGroup * pGroup )
	{
		pGroup->GetAgentGroupData().getAgentData( selectedAgent, _data );
	}
};





}	// namespace OpenSteer
#endif
