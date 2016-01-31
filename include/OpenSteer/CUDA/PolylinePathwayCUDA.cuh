#ifndef OPENSTEER_POLYLINEPATHWAYCUDA_CUH
#define OPENSTEER_POLYLINEPATHWAYCUDA_CUH

#include "CUDAGlobals.cuh"

#include "dev_vector.cuh"
#include <vector>

namespace OpenSteer
{

	class PolylinePathwayCUDA
	{
	private:
		// Device vectors.
		dev_vector< float3 >	m_dvPoints;
		dev_vector< float3 >	m_dvNormals;
		dev_vector< float >		m_dvLengths;

		// Host vectors.
		std::vector< float3 >	m_hvPoints;
		std::vector< float3 >	m_hvNormals;
		std::vector< float >	m_hvLengths;

		float					m_fTotalPathLength;
		float					m_fRadius;
		bool					m_bCyclic;
		uint					m_nPointCount;

		void syncDevice( void )
		{
			m_dvPoints = m_hvPoints;
			m_dvNormals = m_hvNormals;
			m_dvLengths = m_hvLengths;
		}

	public:
		PolylinePathwayCUDA( std::vector< float3 > const& points, float const radius, bool const cyclic );

		~PolylinePathwayCUDA( void );

		void Initialize(	std::vector< float3 > const&	points,
							float const						radius,
							bool const						cyclic
							);

		// Accessors for device data.
		float3 *						pdPoints( void )				{ return m_dvPoints.begin(); }
		float3 *						pdNormals( void )				{ return m_dvNormals.begin(); }
		float *							pdLengths( void )				{ return m_dvLengths.begin(); }

		// Accesors for host data.
		std::vector< float3 > const&	hvPoints( void ) const			{ return m_hvPoints; }
		std::vector< float3 > const&	hvNormals( void ) const			{ return m_hvNormals; }
		std::vector< float > const&		hvLengths( void ) const			{ return m_hvLengths; }

		uint const&						numPoints( void ) const			{ return m_nPointCount; }
		float const&					radius( void ) const			{ return m_fRadius; }
		bool const&						cyclic( void ) const			{ return m_bCyclic; }
		float const&					totalPathLength( void ) const	{ return m_fTotalPathLength; }
	};	// class PolylinePathwayCUDA
}	// namespace OpenSteer


#endif
