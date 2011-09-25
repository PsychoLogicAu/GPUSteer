#ifndef OPENSTEER_WALLDATA_CUH
#define OPENSTEER_WALLDATA_CUH

#include "CUDAGlobals.cuh"

#include "dev_vector.cuh"
#include <vector>

#include "KNNBinData.cuh"

namespace OpenSteer
{
class WallData
{
private:
	dev_vector< float3 >	m_dvLineStart;		// Start points of the line segments.
	dev_vector< float3 >	m_dvLineMid;		// Mid points of the line segments.
	dev_vector< float3 >	m_dvLineEnd;		// End points of the line segments.
	dev_vector< float3 >	m_dvLineNormal;		// Normals of the line segments.

	std::vector< float3 >	m_hvLineStart;
	std::vector< float3 >	m_hvLineMid;
	std::vector< float3 >	m_hvLineEnd;
	std::vector< float3 >	m_hvLineNormal;

	void syncDevice( void );

public:
	WallData( void );
	~WallData( void )
	{}

	void SplitWalls( std::vector< bin_cell > const& cells );

	float3 *	pdLineStart( void )		{ return m_dvLineStart.begin(); }
	float3 *	pdLineMid( void )		{ return m_dvLineMid.begin(); }
	float3 *	pdLineEnd( void )		{ return m_dvLineEnd.begin(); }
	float3 *	pdLineNormal( void )	{ return m_dvLineNormal.begin(); }


};	// class WallData
}	// namespace OpenSteer

#endif
