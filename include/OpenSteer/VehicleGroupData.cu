#ifndef OPENSTEER_VEHICLEGROUPDATA_CU
#define OPENSTEER_VEHICLEGROUPDATA_CU

#include <vector>
#include <cuda_runtime.h>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "VehicleData.h"
#include "VectorUtils.cu"

namespace OpenSteer
{
	// Forward declarations.
	struct vehicle_group_data_device;
	struct vehicle_group_data_host;
	struct vehicle_group_const_device;
	struct vehicle_group_const_host;

	typedef struct vehicle_group_data_device
	{
		// LocalSpace
		thrust::device_vector<float3>	dvSide;		// Side vectors
		thrust::device_vector<float3>	dvUp;		// Up vectors
		thrust::device_vector<float3>	dvForward;	// Forward vector
		thrust::device_vector<float3>	dvPosition;	// Current position
		thrust::device_vector<float3>	dvSteering;	// Steering vector
		// SimpleVehicle
		thrust::device_vector<float>	dvSpeed;	// Current speed
		/// Copy the vehicle_group_data_host structure into this one.
		vehicle_group_data_device & operator=( vehicle_group_data_host const& hvd );
		void clear( void )
		{
			dvSide.clear();
			dvUp.clear();
			dvForward.clear();
			dvPosition.clear();
			dvSteering.clear();
			dvSpeed.clear();
		}

		// Get raw device pointers to the different data elements.
		float3 *	dpSide( void )		{ return thrust::raw_pointer_cast( &dvSide[0] ); }
		float3 *	dpUp( void )		{ return thrust::raw_pointer_cast( &dvUp[0] ); }
		float3 *	dpForward( void )	{ return thrust::raw_pointer_cast( &dvForward[0] ); }
		float3 *	dpPosition( void )	{ return thrust::raw_pointer_cast( &dvPosition[0] ); }
		float3 *	dpSteering( void )	{ return thrust::raw_pointer_cast( &dvSteering[0] ); }
		float *		dpSpeed( void )		{ return thrust::raw_pointer_cast( &dvSpeed[0] ); }
	} VehicleGroupDataDevice;

	typedef struct vehicle_group_data_host
	{
		// LocalSpace
		thrust::host_vector<float3>		hvSide;		// Side vectors
		thrust::host_vector<float3>		hvUp;		// Up vectors
		thrust::host_vector<float3>		hvForward;	// Forward vector
		thrust::host_vector<float3>		hvPosition;	// Current position
		thrust::host_vector<float3>		hvSteering;	// Steering vector
		// SimpleVehicle
		thrust::host_vector<float>		hvSpeed;	// Current speed

		/// Copy the vehicle_group_data_device structure into this one.
		vehicle_group_data_host & operator=( vehicle_group_data_device const& dvd );

		void clear( void )
		{
			hvSide.clear();
			hvUp.clear();
			hvForward.clear();
			hvPosition.clear();
			hvSteering.clear();
			hvSpeed.clear();
		}

		/// Adds a vehicle_data structure to this structure.
		void AddVehicle( vehicle_data const& vd )
		{
			hvSide.push_back( vd.side );
			hvUp.push_back( vd.up );
			hvForward.push_back( vd.forward );
			hvPosition.push_back( vd.position );
			hvSteering.push_back( vd.steering );
			hvSpeed.push_back( vd.speed );
		}
		/// Removes the vehicle_data structure at index.
		void RemoveVehicle( size_t const index )
		{
			hvSide.erase( hvSide.begin() + index );
			hvUp.erase( hvUp.begin() + index );
			hvForward.erase( hvForward.begin() + index );
			hvPosition.erase( hvPosition.begin() + index );
			hvSteering.erase( hvSteering.begin() + index );
			hvSpeed.erase( hvSpeed.begin() + index );
		}
		/// Get the data for the vehicle_const structure at index.
		void GetVehicleData( size_t const index, vehicle_data & vd )
		{
			vd.side		= hvSide[ index ];
			vd.up		= hvUp[ index ];
			vd.forward	= hvForward[ index ];
			vd.position	= hvPosition[ index ];
			vd.steering	= hvSteering[ index ];
			vd.speed	= hvSpeed[ index ];
		}
	} VehicleGroupDataHost;

	typedef struct vehicle_group_const_device
	{
		// Device data.
		thrust::device_vector<id_type>		dvId;
		// SimpleVehicle
		thrust::device_vector<float>		dvMaxForce;
		thrust::device_vector<float>		dvMaxSpeed;
		thrust::device_vector<float>		dvMass;
		thrust::device_vector<float>		dvRadius;
		/// Copy the vehicle_group_const_host structure into this one.
		vehicle_group_const_device & operator=( vehicle_group_const_host const& hvc );

		void clear( void )
		{
			dvId.clear();
			dvMaxForce.clear();
			dvMaxSpeed.clear();
			dvMass.clear();
			dvRadius.clear();
		}

		// Get raw device pointers to the different data elements.
		id_type *	dpId( void )		{ return thrust::raw_pointer_cast( &dvId[0] ); }
		float *		dpMaxForce( void )	{ return thrust::raw_pointer_cast( &dvMaxForce[0] ); }
		float *		dpMaxSpeed( void )	{ return thrust::raw_pointer_cast( &dvMaxSpeed[0] ); }
		float *		dpMass( void )		{ return thrust::raw_pointer_cast( &dvMass[0] ); }
		float *		dpRadius( void )	{ return thrust::raw_pointer_cast( &dvRadius[0] ); }
	} VehicleGroupConstDevice;

	typedef struct vehicle_group_const_host
	{
		// Host data.
		thrust::host_vector<id_type>		hvId;
		// SimpleVehicle
		thrust::host_vector<float>			hvMaxForce;
		thrust::host_vector<float>			hvMaxSpeed;
		thrust::host_vector<float>			hvMass;
		thrust::host_vector<float>			hvRadius;

		/// Copy the vehicle_group_const_device structure into this one.
		vehicle_group_const_host & operator=( vehicle_group_const_device const& dvc );

		void clear( void )
		{
			hvId.clear();
			hvMaxForce.clear();
			hvMaxSpeed.clear();
			hvMass.clear();
			hvRadius.clear();
		}

		/// Adds a vehicle_const structure.
		void AddVehicle( vehicle_const const& vc )
		{
			hvId.push_back( vc.id );
			hvMaxForce.push_back( vc.maxForce );
			hvMaxSpeed.push_back( vc.maxSpeed );
			hvMass.push_back( vc.mass );
			hvRadius.push_back( vc.radius );
		}
		/// Removes the vehicle_const structure at index.
		void RemoveVehicle( size_t const index )
		{
			hvId.erase( hvId.begin() + index );
			hvMaxForce.erase( hvMaxForce.begin() + index );
			hvMaxSpeed.erase( hvMaxSpeed.begin() + index );
			hvMass.erase( hvMass.begin() + index );
			hvRadius.erase( hvRadius.begin() + index );
		}
		/// Get the data for the vehicle_const structure at index.
		void GetVehicleData( size_t const index, vehicle_const & vc )
		{
			vc.id		= hvId[ index ];
			vc.mass		= hvMass[ index ];
			vc.maxForce	= hvMaxForce[ index ];
			vc.maxSpeed	= hvMaxSpeed[ index ];
			vc.radius	= hvRadius[ index ];
		}
	} VehicleGroupConstHost;

	vehicle_group_data_device & vehicle_group_data_device::operator=( vehicle_group_data_host const& hvd )
	{
		dvSide		= hvd.hvSide;
		dvUp		= hvd.hvUp;
		dvForward	= hvd.hvForward;
		dvPosition	= hvd.hvPosition;
		dvSteering	= hvd.hvSteering;
		dvSpeed		= hvd.hvSpeed;
		return *this;
	}

	vehicle_group_data_host & vehicle_group_data_host::operator=( vehicle_group_data_device const& dvd )
	{
		hvSide		= dvd.dvSide;
		hvUp		= dvd.dvUp;
		hvForward	= dvd.dvForward;
		hvPosition	= dvd.dvPosition;
		hvSteering	= dvd.dvSteering;
		hvSpeed		= dvd.dvSpeed;
		return *this;
	}

	vehicle_group_const_device & vehicle_group_const_device::operator=( vehicle_group_const_host const& hvc )
	{
		dvId		= hvc.hvId;
		dvMaxForce	= hvc.hvMaxForce;
		dvMaxSpeed	= hvc.hvMaxSpeed;
		dvMass		= hvc.hvMass;
		dvRadius	= hvc.hvRadius;
		return *this;
	}

	vehicle_group_const_host & vehicle_group_const_host::operator=( vehicle_group_const_device const& dvc )
	{
		hvId		= dvc.dvId;
		hvMaxForce	= dvc.dvMaxForce;
		hvMaxSpeed	= dvc.dvMaxSpeed;
		hvMass		= dvc.dvMass;
		hvRadius	= dvc.dvRadius;
		return *this;
	}

	/// Compute the velocity of the vehicle at index i.
	static inline __host__ __device__ float3 velocity( size_t const i, float3 const* pForward, float const* pSpeed )
	{
		return float3_scalar_multiply( pForward[i], pSpeed[i] );
	}
	/// Compute the predicted position of the vehicle at index i at time (current + fPredictionTime).
	static inline __host__ __device__ float3 predictFuturePosition( size_t const i, float3 const* pPosition, float3 const* pForward, float const* pSpeed, float const fPredictionTime )
	{
		return float3_add( pPosition[i], float3_scalar_multiply( velocity( i, pForward, pSpeed ), fPredictionTime ));
	}

	/*
	typedef struct vehicle_data {
		// LocalSpace
		float3  side;					// Side vector
		float3  up;						// Up vector
		float3  forward;				// Forward vector
		float3  position;				// Current position
		float3	steering;				// Steering vector
	    
		// SimpleVehicle
		float   speed;					// Current speed

		__host__ __device__ float3 predictFuturePosition(const float predictionTime) const { return float3_add(position, float3_scalar_multiply(velocity(), predictionTime)); }
		__host__ __device__ float3 velocity(void) const { return float3_scalar_multiply(forward, speed); }
	} VehicleData;

	typedef struct vehicle_const {
		unsigned int id;

		// SimpleVehicle
		float   maxForce;
		float   maxSpeed;
		float   mass;
		float   radius;
	} VehicleConst;

	typedef std::vector<VehicleData> DataVec;
	typedef DataVec::iterator DataVecIt;

	typedef std::vector<VehicleConst> ConstVec;
	typedef ConstVec::iterator ConstVecIt;
	*/
}//namespace OpenSteer
#endif // VEHICLE_DATA_H