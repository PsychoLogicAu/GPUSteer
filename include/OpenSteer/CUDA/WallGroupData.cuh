#ifndef OPENSTEER_WALLDATA_CUH
#define OPENSTEER_WALLDATA_CUH

#include "CUDAGlobals.cuh"

#include "dev_vector.cuh"
#include <vector>

#include "KNNBinData.cuh"

namespace OpenSteer
{
class WallGroupData
{
private:
	dev_vector< float4 >	m_dvLineStart;		// Start points of the line segments.
	dev_vector< float4 >	m_dvLineMid;		// Mid points of the line segments.
	dev_vector< float4 >	m_dvLineEnd;		// End points of the line segments.
	dev_vector< float4 >	m_dvLineNormal;		// Normals of the line segments.

	std::vector< float4 >	m_hvLineStart;
	std::vector< float4 >	m_hvLineMid;
	std::vector< float4 >	m_hvLineEnd;
	std::vector< float4 >	m_hvLineNormal;

	void syncDevice( void );

	bool intersects(	float3 const& start, float3 const& end,				// Start and end of line segment.
						float3 const& cellMin, float3 const& cellMax,		// Min and max of cell.
						float3 & intersectPoint								// The point of intersection with the cell.
						);

	uint					m_nCount;

public:
	WallGroupData( void );
	~WallGroupData( void )
	{}

	void SplitWalls( std::vector< bin_cell > const& cells );
	bool LoadFromFile( char const* szFilename );

	float4 *	pdLineStart( void )		{ return m_dvLineStart.begin(); }
	float4 *	pdLineMid( void )		{ return m_dvLineMid.begin(); }
	float4 *	pdLineEnd( void )		{ return m_dvLineEnd.begin(); }
	float4 *	pdLineNormal( void )	{ return m_dvLineNormal.begin(); }

	std::vector< float4 > const&	hvLineStart( void ) const	{ return m_hvLineStart; }
	std::vector< float4 > const&	hvLineMid( void ) const		{ return m_hvLineMid; }
	std::vector< float4 > const&	hvLineEnd( void ) const		{ return m_hvLineEnd; }
	std::vector< float4 > const&	hvLineNormal( void ) const	{ return m_hvLineNormal; }

	uint const&						size( void ) const			{ return m_nCount; }
};	// class WallGroupData
}	// namespace OpenSteer

#endif
