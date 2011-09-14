#ifndef OPENSTEER_WALLDATA_CUH
#define OPENSTEER_WALLDATA_CUH

#include "CUDAGlobals.cuh"

#include "dev_vector.cuh"
#include <vector>

class wall_data
{
private:
	dev_vector< float3 >	m_dvLineStart;		// Start points of the line segments.
	dev_vector< float3 >	m_dvLineEnd;		// End points of the line segments.
	dev_vector< float3 >	m_dvLineNormal;		// Normals of the line segments.

	std::vector< float3 >	m_hvLineStart;
	std::vector< float3 >	m_hvLineEnd;
	std::vector< float3 >	m_hvLineNormal;




public:
	wall_data( void );
	~wall_data( void );


};	// class wall_data


#endif
