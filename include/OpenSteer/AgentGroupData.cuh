#ifndef OPENSTEER_AGENTGROUPDATA_CUH
#define OPENSTEER_AGENTGROUPDATA_CUH

#include <vector>
#include <cuda_runtime.h>
#include <vector_types.h>

#include <vector>

#include "CUDA\CUDAGlobals.cuh"

#include "AgentData.h"
#include "VectorUtils.cuh"

#include "CUDA\dev_vector.cuh"

namespace OpenSteer
{
class agent_group_data
{
	friend class AgentGroup;
private:
	bool	m_bSyncHost;
	bool	m_bSyncDevice;
	size_t	m_nSize;

	//
	//	Device vectors
	//
	dev_vector<id_type>		m_dvID;						// Agent IDs

	dev_vector<float3>		m_dvSide;					// Side vectors
	dev_vector<float3>		m_dvUp;						// Up vectors
	dev_vector<float4>		m_dvDirection;				// Direction vectors
	dev_vector<float4>		m_dvPosition;				// Position vectors and Radii

	dev_vector<float4>		m_dvSteering;				// Steering vectors and MaxForce
	dev_vector<float>		m_dvSpeed;					// Current speed

	dev_vector<float>		m_dvMaxSpeed;
	dev_vector<float>		m_dvMaxForce;
	dev_vector<float>		m_dvRadius;
	dev_vector<float>		m_dvMass;					// Agent mass
	
	// Bitmask
	dev_vector<uint>		m_dvAppliedKernels;			// Bitmask of applied kernels this update.

	//
	//	Host vectors
	//
	std::vector<id_type>	m_hvID;						// Agent IDs

	std::vector<float3>		m_hvSide;					// Side vectors
	std::vector<float3>		m_hvUp;						// Up vectors
	std::vector<float4>		m_hvDirection;				// Direction vectors
	std::vector<float4>		m_hvPosition;				// Position vectors

	std::vector<float4>		m_hvSteering;				// Steering vectors
	std::vector<float>		m_hvSpeed;					// Current speed

	std::vector<float>		m_hvMaxSpeed;
	std::vector<float>		m_hvMaxForce;
	std::vector<float>		m_hvRadius;
	std::vector<float>		m_hvMass;					// Agent mass

	// Bitmask
	std::vector<uint>		m_hvAppliedKernels;			// Bitmask of applied kernels this update.

public:
	agent_group_data( void )
		:	m_nSize( 0 ),
			m_bSyncDevice( false ),
			m_bSyncHost( false )
	{ }
	~agent_group_data( void )	{ }

	// Accessors for the device data.
	id_type *	pdID( void )					{ return m_dvID.begin(); }

	float3 *	pdSide( void )					{ return m_dvSide.begin(); }
	float3 *	pdUp( void )					{ return m_dvUp.begin(); }
	float4 *	pdDirection( void )				{ return m_dvDirection.begin(); }
	float4 *	pdPosition( void )				{ return m_dvPosition.begin(); }

	float4 *	pdSteering( void )				{ return m_dvSteering.begin(); }
	float *		pdSpeed( void )					{ return m_dvSpeed.begin(); }

	float *		pdMaxSpeed( void )				{ return m_dvMaxSpeed.begin(); }
	float *		pdMaxForce( void )				{ return m_dvMaxForce.begin(); }
	float *		pdRadius( void )				{ return m_dvRadius.begin(); }
	float *		pdMass( void )					{ return m_dvMass.begin(); }

	uint *		pdAppliedKernels( void )		{ return m_dvAppliedKernels.begin(); }

	// Accessors for the host data.
	std::vector<id_type> const& hvID( void ) const					{ return m_hvID; }
	std::vector<id_type> & hvID( void )								{ m_bSyncDevice = true; return m_hvID; }

	std::vector<float3> const& hvSide( void ) const					{ return m_hvSide; }
	std::vector<float3> & hvSide( void )							{ m_bSyncDevice = true; return m_hvSide; }
	std::vector<float3> const& hvUp( void ) const					{ return m_hvUp; }
	std::vector<float3> & hvUp( void )								{ m_bSyncDevice = true; return m_hvUp; }
	std::vector<float4> const& hvDirection( void ) const			{ return m_hvDirection; }
	std::vector<float4> & hvDirection( void )						{ m_bSyncDevice = true; return m_hvDirection; }
	std::vector<float4> const& hvPosition( void ) const				{ return m_hvPosition; }
	std::vector<float4> & hvPosition( void )						{ m_bSyncDevice = true; return m_hvPosition; }

	std::vector<float4> const& hvSteering( void ) const				{ return m_hvSteering; }
	std::vector<float4> & hvSteering( void )						{ m_bSyncDevice = true; return m_hvSteering; }
	std::vector<float> const& hvSpeed( void ) const					{ return m_hvSpeed; }
	std::vector<float> & hvSpeed( void )							{ m_bSyncDevice = true; return m_hvSpeed; }

	std::vector<float> const& hvMaxSpeed( void ) const				{ return m_hvMaxSpeed; }
	std::vector<float> & hvMaxSpeed( void )							{ m_bSyncDevice = true; return m_hvMaxSpeed; }
	std::vector<float> const& hvMaxForce( void ) const				{ return m_hvMaxForce; }
	std::vector<float> & hvMaxForce( void )							{ m_bSyncDevice = true; return m_hvMaxForce; }
	std::vector<float> const& hvRadius( void ) const				{ return m_hvRadius; }
	std::vector<float> & hvRadius( void )							{ m_bSyncDevice = true; return m_hvRadius; }
	std::vector<float> const& hvMass( void ) const					{ return m_hvMass; }
	std::vector<float> & hvMass( void )								{ m_bSyncDevice = true; return m_hvMass; }

	std::vector<uint> const& hvAppliedKernels( void ) const			{ return m_hvAppliedKernels; }
	std::vector<uint> & hvAppliedKernels( void )					{ m_bSyncDevice = true; return m_hvAppliedKernels; }

	size_t size( void ) const	{ return m_nSize; }

	/// Adds an agent from a vehicle_data structure.
	void addAgent( agent_data const& ad )
	{
		m_hvID.push_back( ad.id );

		m_hvSide.push_back( ad.side );
		m_hvUp.push_back( ad.up );
		m_hvDirection.push_back( ad.direction );
		m_hvPosition.push_back( ad.position );

		m_hvSteering.push_back( ad.steering );
		m_hvSpeed.push_back( ad.speed );

		m_hvMaxSpeed.push_back( ad.maxSpeed );
		m_hvMaxForce.push_back( ad.maxForce );
		m_hvRadius.push_back( ad.radius );
		m_hvMass.push_back( ad.mass );

		m_hvAppliedKernels.push_back( ad.appliedKernels );

		m_nSize++;
		m_bSyncDevice = true;
	}
	/// Removes the vehicle structure at index.
	void removeAgent( size_t const index )
	{
		if( index < m_nSize )
		{
			m_hvID.erase( m_hvID.begin() + index );

			m_hvSide.erase( m_hvSide.begin() + index );
			m_hvUp.erase( m_hvUp.begin() + index );
			m_hvDirection.erase( m_hvDirection.begin() + index );
			m_hvPosition.erase( m_hvPosition.begin() + index );

			m_hvSteering.erase( m_hvSteering.begin() + index );
			m_hvSpeed.erase( m_hvSpeed.begin() + index );

			m_hvMaxSpeed.erase( m_hvMaxSpeed.begin() + index );
			m_hvMaxForce.erase( m_hvMaxForce.begin() + index );
			m_hvRadius.erase( m_hvRadius.begin() + index );
			m_hvMass.erase( m_hvMass.begin() + index );

			m_hvAppliedKernels.erase( m_hvAppliedKernels.begin() + index );
			
			m_nSize--;
			m_bSyncDevice = true;
		}
	}
	/// Get the data for the vehicle_const structure at index.
	void getAgentData( size_t const index, agent_data & ad )
	{
		if( index < m_nSize )
		{
			syncHost();

			ad.id					= m_hvID[ index ];

			ad.side					= m_hvSide[ index ];
			ad.up					= m_hvUp[ index ];
			ad.direction			= m_hvDirection[ index ];
			ad.position				= m_hvPosition[ index ];

			ad.steering				= m_hvSteering[ index ];
			ad.speed				= m_hvSpeed[ index ];

			ad.maxSpeed				= m_hvMaxSpeed[ index ];
			ad.maxForce				= m_hvMaxForce[ index ];
			ad.radius				= m_hvRadius[ index ];
			ad.mass					= m_hvMass[ index ];

			ad.appliedKernels		= m_hvAppliedKernels[ index ];
		}
	}

	void syncHost( void )
	{
		if( m_bSyncHost )
		{
			cudaMemcpy( &m_hvID[0], pdID(), m_nSize * sizeof(id_type), cudaMemcpyDeviceToHost );

			cudaMemcpy( &m_hvSide[0], pdSide(), m_nSize * sizeof(float3), cudaMemcpyDeviceToHost );
			cudaMemcpy( &m_hvUp[0], pdUp(), m_nSize * sizeof(float3), cudaMemcpyDeviceToHost );
			cudaMemcpy( &m_hvDirection[0], pdDirection(), m_nSize * sizeof(float4), cudaMemcpyDeviceToHost );
			cudaMemcpy( &m_hvPosition[0], pdPosition(), m_nSize * sizeof(float4), cudaMemcpyDeviceToHost );

			cudaMemcpy( &m_hvSteering[0], pdSteering(), m_nSize * sizeof(float4), cudaMemcpyDeviceToHost );
			cudaMemcpy( &m_hvSpeed[0], pdSpeed(), m_nSize * sizeof(float), cudaMemcpyDeviceToHost );

			cudaMemcpy( &m_hvMaxSpeed[0], pdMaxSpeed(), m_nSize * sizeof(float), cudaMemcpyDeviceToHost );
			cudaMemcpy( &m_hvMaxForce[0], pdMaxForce(), m_nSize * sizeof(float), cudaMemcpyDeviceToHost );
			cudaMemcpy( &m_hvRadius[0], pdRadius(), m_nSize * sizeof(float), cudaMemcpyDeviceToHost );
			cudaMemcpy( &m_hvMass[0], pdMass(), m_nSize * sizeof(float), cudaMemcpyDeviceToHost );

			cudaMemcpy( &m_hvAppliedKernels[0], pdAppliedKernels(), m_nSize * sizeof(uint), cudaMemcpyDeviceToHost );

			m_bSyncHost = false;
		}
	}

	void syncDevice( void )
	{
		if( m_bSyncDevice )
		{
			m_dvID = m_hvID;

			m_dvSide = m_hvSide;
			m_dvUp = m_hvUp;
			m_dvDirection = m_hvDirection;
			m_dvPosition = m_hvPosition;

			m_dvSteering = m_hvSteering;
			m_dvSpeed = m_hvSpeed;

			m_dvMaxSpeed = m_hvMaxSpeed;
			m_dvMaxForce = m_hvMaxForce;
			m_dvRadius = m_hvRadius;
			m_dvMass = m_hvMass;

			m_dvAppliedKernels = m_hvAppliedKernels;

			m_bSyncDevice = false;
		}
	}

	void clear( void )
	{
		m_nSize = 0;
		m_bSyncHost = false;
		m_bSyncDevice = false;

		m_dvID.clear();

		m_dvSide.clear();
		m_dvUp.clear();
		m_dvDirection.clear();
		m_dvPosition.clear();

		m_dvSteering.clear();
		m_dvSpeed.clear();

		m_dvMaxSpeed.clear();
		m_dvMaxForce.clear();
		m_dvRadius.clear();
		m_dvMass.clear();

		m_dvAppliedKernels.clear();

		m_hvID.clear();

		m_hvSide.clear();
		m_hvUp.clear();
		m_hvDirection.clear();
		m_hvPosition.clear();

		m_hvSteering.clear();
		m_hvSpeed.clear();

		m_hvMaxSpeed.clear();
		m_hvMaxForce.clear();
		m_hvRadius.clear();
		m_hvMass.clear();

		m_hvAppliedKernels.clear();
	}
};	// class agent_group_data
typedef agent_group_data AgentGroupData;

/// Compute the velocity of the vehicle at index i.
static inline __host__ __device__ float3 velocity( size_t const i, float3 const* pDirection, float const* pSpeed )
{
	return float3_scalar_multiply( pDirection[i], pSpeed[i] );
}
/// Compute the predicted position of the vehicle at index i at time (current + fPredictionTime).
static inline __host__ __device__ float3 predictFuturePosition( size_t const i, float3 const* pPosition, float3 const* pForward, float const* pSpeed, float const fPredictionTime )
{
	return float3_add( pPosition[i], float3_scalar_multiply( velocity( i, pForward, pSpeed ), fPredictionTime ));
}

}	// namespace OpenSteer
#endif	// OPENSTEER_AGENTGROUPDATA_CUH
