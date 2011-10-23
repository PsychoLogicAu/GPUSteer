#include "PluginCommon.h"

using namespace OpenSteer;

void OpenSteer::randomizeStartingPositionAndHeading2D( float4 & position, float3 & up, float4 & forward, float3 & side, float const minRadius, float const maxRadius, float3 const& startPosition )
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

void OpenSteer::randomizeStartingPositionAndHeading3D( float4 & position, float3 & up, float4 & forward, float3 & side, float const minRadius, float const maxRadius, float3 const& startPosition )
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