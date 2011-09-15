#ifndef OPENSTEER_VEHICLEGROUPDATA_CUH
#define OPENSTEER_VEHICLEGROUPDATA_CUH

#include <vector>
#include <cuda_runtime.h>
#include <vector_types.h>

#include <vector>

#include "VehicleData.h"
#include "VectorUtils.cuh"

#include "CUDA\dev_vector.cuh"

namespace OpenSteer
{
// Forward declarations.
class vehicle_group_data;
class vehicle_group_const;

class vehicle_group_data
{
	friend class VehicleGroup;
private:
	bool	m_bSyncHost;
	bool	m_bSyncDevice;
	size_t	m_nSize;

	//
	//	Device vectors
	//
	// LocalSpace
	dev_vector<float3>		m_dvSide;		// Side vectors
	dev_vector<float3>		m_dvUp;			// Up vectors
	dev_vector<float3>		m_dvForward;	// Forward vector
	dev_vector<float3>		m_dvPosition;	// Current position
	dev_vector<float3>		m_dvSteering;	// Steering vector
	// SimpleVehicle
	dev_vector<float>		m_dvSpeed;		// Current speed

	//
	//	Host vectors
	//
	// LocalSpace
	std::vector<float3>		m_hvSide;		// Side vectors
	std::vector<float3>		m_hvUp;			// Up vectors
	std::vector<float3>		m_hvForward;	// Forward vector
	std::vector<float3>		m_hvPosition;	// Current position
	std::vector<float3>		m_hvSteering;	// Steering vector
	// SimpleVehicle
	std::vector<float>		m_hvSpeed;		// Current speed

public:
	vehicle_group_data( void )
		:	m_nSize( 0 ),
			m_bSyncDevice( false ),
			m_bSyncHost( false )
	{ }
	~vehicle_group_data( void )	{ }

	// Accessors for the device data.
	float3 *	pdSide( void )		{ return m_dvSide.begin(); }
	float3 *	pdUp( void )		{ return m_dvUp.begin(); }
	float3 *	pdForward( void )	{ return m_dvForward.begin(); }
	float3 *	pdPosition( void )	{ return m_dvPosition.begin(); }
	float3 *	pdSteering( void )	{ return m_dvSteering.begin(); }
	float *		pdSpeed( void )		{ return m_dvSpeed.begin(); }

	// Accessors for the host data.
	std::vector<float3> const& hvSide( void ) const			{ return m_hvSide; }
	std::vector<float3> & hvSide( void )					{ m_bSyncDevice = true; return m_hvSide; }
	std::vector<float3> const& hvUp( void ) const			{ return m_hvUp; }
	std::vector<float3> & hvUp( void )						{ m_bSyncDevice = true; return m_hvUp; }
	std::vector<float3> const& hvForward( void ) const		{ return m_hvForward; }
	std::vector<float3> & hvForward( void )					{ m_bSyncDevice = true; return m_hvForward; }
	std::vector<float3> const& hvPosition( void ) const		{ return m_hvPosition; }
	std::vector<float3> & hvPosition( void )				{ m_bSyncDevice = true; return m_hvPosition; }
	std::vector<float3> const& hvSteering( void ) const		{ return m_hvSteering; }
	std::vector<float3> & hvSteering( void )				{ m_bSyncDevice = true; return m_hvSteering; }
	std::vector<float> const& hvSpeed( void ) const	{ return m_hvSpeed; }
	std::vector<float> & hvSpeed( void )					{ m_bSyncDevice = true; return m_hvSpeed; }

	size_t size( void ) const	{ return m_nSize; }

	/// Adds an agent from a vehicle_data structure.
	void addVehicle( vehicle_data const& vd )
	{
		m_hvSide.push_back( vd.side );
		m_hvUp.push_back( vd.up );
		m_hvForward.push_back( vd.forward );
		m_hvPosition.push_back( vd.position );
		m_hvSteering.push_back( vd.steering );
		m_hvSpeed.push_back( vd.speed );

		m_nSize++;
		m_bSyncDevice = true;
	}
	/// Removes the vehicle structure at index.
	void removeVehicle( size_t const index )
	{
		if( index < m_nSize )
		{
			m_hvSide.erase( m_hvSide.begin() + index );
			m_hvUp.erase( m_hvUp.begin() + index );
			m_hvForward.erase( m_hvForward.begin() + index );
			m_hvPosition.erase( m_hvPosition.begin() + index );
			m_hvSteering.erase( m_hvSteering.begin() + index );
			m_hvSpeed.erase( m_hvSpeed.begin() + index );
			
			m_nSize--;
			m_bSyncDevice = true;
		}
	}
	/// Get the data for the vehicle_const structure at index.
	void getVehicleData( size_t const index, vehicle_data & vd )
	{
		if( index < m_nSize )
		{
			syncHost();

			vd.side		= m_hvSide[ index ];
			vd.up		= m_hvUp[ index ];
			vd.forward	= m_hvForward[ index ];
			vd.position	= m_hvPosition[ index ];
			vd.steering	= m_hvSteering[ index ];
			vd.speed	= m_hvSpeed[ index ];
		}
	}

	void syncHost( void )
	{
		if( m_bSyncHost )
		{
			cudaMemcpy( &m_hvSide[0], pdSide(), m_nSize * sizeof(float3), cudaMemcpyDeviceToHost );
			cudaMemcpy( &m_hvUp[0], pdUp(), m_nSize * sizeof(float3), cudaMemcpyDeviceToHost );
			cudaMemcpy( &m_hvForward[0], pdForward(), m_nSize * sizeof(float3), cudaMemcpyDeviceToHost );
			cudaMemcpy( &m_hvPosition[0], pdPosition(), m_nSize * sizeof(float3), cudaMemcpyDeviceToHost );
			cudaMemcpy( &m_hvSteering[0], pdSteering(), m_nSize * sizeof(float3), cudaMemcpyDeviceToHost );
			cudaMemcpy( &m_hvSpeed[0], pdSpeed(), m_nSize * sizeof(float), cudaMemcpyDeviceToHost );

			m_bSyncHost = false;
		}
	}

	void syncDevice( void )
	{
		if( m_bSyncDevice )
		{
			m_dvSide = m_hvSide;
			m_dvUp = m_hvUp;
			m_dvForward = m_hvForward;
			m_dvPosition = m_hvPosition;
			m_dvSteering = m_hvSteering;
			m_dvSpeed = m_hvSpeed;

			m_bSyncDevice = false;
		}
	}

	void clear( void )
	{
		m_nSize = 0;
		m_bSyncHost = false;
		m_bSyncDevice = false;

		m_dvSide.clear();
		m_dvUp.clear();
		m_dvForward.clear();
		m_dvPosition.clear();
		m_dvSteering.clear();
		m_dvSpeed.clear();

		m_hvSide.clear();
		m_hvUp.clear();
		m_hvForward.clear();
		m_hvPosition.clear();
		m_hvSteering.clear();
		m_hvSpeed.clear();
	}
};
typedef vehicle_group_data VehicleGroupData;


class vehicle_group_const
{
	friend class VehicleGroup;
private:
	//
	// Device data.
	//
	dev_vector<id_type>			m_dvId;
	// SimpleVehicle
	dev_vector<float>			m_dvMaxForce;
	dev_vector<float>			m_dvMaxSpeed;
	dev_vector<float>			m_dvMass;
	dev_vector<float>			m_dvRadius;

	//
	// Host data.
	//
	std::vector<id_type>		m_hvId;
	// SimpleVehicle
	std::vector<float>			m_hvMaxForce;
	std::vector<float>			m_hvMaxSpeed;
	std::vector<float>			m_hvMass;
	std::vector<float>			m_hvRadius;

	size_t						m_nSize;
	bool						m_bSyncDevice;
	bool						m_bSyncHost;

public:
	vehicle_group_const( void )
		:	m_nSize( 0 ),
			m_bSyncDevice( false ),
			m_bSyncHost( false )
	{ }

	~vehicle_group_const( void )	{ }

	std::vector<id_type> const& hvId( void ) const		{ return m_hvId; }
	std::vector<id_type> hvId( void )						{ m_bSyncDevice = true; return m_hvId; }
	std::vector<float> const& hvMaxForce( void ) const	{ return m_hvMaxForce; }
	std::vector<float> hvMaxForce( void )				{ m_bSyncDevice = true; return m_hvMaxForce; }
	std::vector<float> const& hvMaxSpeed( void ) const	{ return m_hvMaxSpeed; }
	std::vector<float> hvMaxSpeed( void )				{ m_bSyncDevice = true; return m_hvMaxSpeed; }
	std::vector<float> const& hvMass( void ) const		{ return m_hvMass; }
	std::vector<float> hvMass( void )					{ m_bSyncDevice = true; return m_hvMass; }
	std::vector<float> const& hvRadius( void ) const	{ return m_hvRadius; }
	std::vector<float> hvRadius( void )					{ m_bSyncDevice = true; return m_hvRadius; }

	size_t size( void ) const	{ return m_nSize; }

	void syncHost( void )
	{
		if( m_bSyncHost )
		{
			cudaMemcpy( &m_hvId[0], pdId(), m_nSize * sizeof(id_type), cudaMemcpyDeviceToHost );
			cudaMemcpy( &m_hvMaxForce[0], pdMaxForce(), m_nSize * sizeof(float), cudaMemcpyDeviceToHost );
			cudaMemcpy( &m_hvMaxSpeed[0], pdMaxSpeed(), m_nSize * sizeof(float), cudaMemcpyDeviceToHost );
			cudaMemcpy( &m_hvMass[0], pdMass(), m_nSize * sizeof(float), cudaMemcpyDeviceToHost );
			cudaMemcpy( &m_hvRadius[0], pdRadius(), m_nSize * sizeof(float), cudaMemcpyDeviceToHost );

			m_bSyncHost = false;
		}
	}

	void syncDevice( void )
	{
		if( m_bSyncDevice )
		{
			m_dvId = m_hvId;
			m_dvMaxForce = m_hvMaxForce;
			m_dvMaxSpeed = m_hvMaxSpeed;
			m_dvMass = m_hvMass;
			m_dvRadius = m_hvRadius;

			m_bSyncDevice = false;
		}
	}

	void clear( void )
	{
		m_nSize = 0;
		m_bSyncHost = false;
		m_bSyncDevice = false;

		m_dvId.clear();
		m_dvMaxForce.clear();
		m_dvMaxSpeed.clear();
		m_dvMass.clear();
		m_dvRadius.clear();

		m_hvId.clear();
		m_hvMaxForce.clear();
		m_hvMaxSpeed.clear();
		m_hvMass.clear();
		m_hvRadius.clear();
	}

	/// Adds a vehicle_const structure.
	void addVehicle( vehicle_const const& vc )
	{
		m_hvId.push_back( vc.id );
		m_hvMaxForce.push_back( vc.maxForce );
		m_hvMaxSpeed.push_back( vc.maxSpeed );
		m_hvMass.push_back( vc.mass );
		m_hvRadius.push_back( vc.radius );

		m_nSize++;
		m_bSyncDevice = true;
	}
	/// Removes the vehicle_const structure at index.
	void removeVehicle( size_t const index )
	{
		m_hvId.erase( m_hvId.begin() + index );
		m_hvMaxForce.erase( m_hvMaxForce.begin() + index );
		m_hvMaxSpeed.erase( m_hvMaxSpeed.begin() + index );
		m_hvMass.erase( m_hvMass.begin() + index );
		m_hvRadius.erase( m_hvRadius.begin() + index );

		m_nSize--;
		m_bSyncDevice = true;
	}
	/// Get the data for the vehicle_const structure at index.
	void getVehicleData( size_t const index, vehicle_const & vc )
	{
		syncHost();

		vc.id		= m_hvId[ index ];
		vc.mass		= m_hvMass[ index ];
		vc.maxForce	= m_hvMaxForce[ index ];
		vc.maxSpeed	= m_hvMaxSpeed[ index ];
		vc.radius	= m_hvRadius[ index ];
	}

	// Get raw device pointers to the different data elements.
	id_type *	pdId( void )		{ return m_dvId.begin(); }
	float *		pdMaxForce( void )	{ return m_dvMaxForce.begin(); }
	float *		pdMaxSpeed( void )	{ return m_dvMaxSpeed.begin(); }
	float *		pdMass( void )		{ return m_dvMass.begin(); }
	float *		pdRadius( void )	{ return m_dvRadius.begin(); }
};
typedef vehicle_group_const VehicleGroupConst;

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

}//namespace OpenSteer
#endif // VEHICLE_DATA_H
