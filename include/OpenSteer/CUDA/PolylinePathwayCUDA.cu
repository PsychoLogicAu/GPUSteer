#include "PolylinePathwayCUDA.cuh"

#include "../VectorUtils.cuh"

using namespace OpenSteer;

PolylinePathwayCUDA::PolylinePathwayCUDA( std::vector< float3 > const& points, float const radius, bool const cyclic )
{
	Initialize( points, radius, cyclic );
}

PolylinePathwayCUDA::~PolylinePathwayCUDA( void )
{
	// Nothing to do.
}

void PolylinePathwayCUDA::Initialize(	std::vector< float3 > const&	points,
										float const						radius,
										bool const						cyclic
										)
{
	// set data members, allocate arrays
	m_fRadius = radius;
	m_bCyclic = cyclic;
	m_nPointCount = points.size();
	m_fTotalPathLength = 0.f;

	if( cyclic )
		m_nPointCount++;

	// Resize the vectors.
	m_hvLengths.resize( m_nPointCount );
	m_hvPoints.resize( m_nPointCount );
	m_hvNormals.resize( m_nPointCount );

	// loop over all points
	for( uint i = 0; i < m_nPointCount; i++ )
	{
		// copy in point locations, closing cycle when appropriate
		const bool closeCycle = cyclic && (i == m_nPointCount - 1);
		const int j = closeCycle ? 0 : i;
		m_hvPoints[i] = points[j];

		// for the end of each segment
		if (i > 0)
		{
			// compute the segment length
			m_hvNormals[i] = float3_subtract( m_hvPoints[i], m_hvPoints[i-1] );
			m_hvLengths[i] = float3_length( m_hvNormals[i] );

			// find the normalized vector parallel to the segment
			m_hvNormals[i] = float3_normalize( m_hvNormals[i] );

			// keep running total of segment lengths
			m_fTotalPathLength += m_hvLengths[i];
		}
	}

	// Send the data to the device.
	syncDevice();
}