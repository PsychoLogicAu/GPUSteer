#ifndef OPENSTEER_VEHICLEDATA_H
#define OPENSTEER_VEHICLEDATA_H

#include <vector>
#include <cuda_runtime.h>

#include "VectorUtils.cuh"

namespace OpenSteer
{
	typedef unsigned int id_type;

	typedef struct vehicle_data
	{
		id_type	id;

		// LocalSpace
		float3	side;					// Side vector
		float3	up;						// Up vector
		float3	forward;				// Forward vector
		float3	position;				// Current position
		float3	steering;				// Steering vector
	    
		// SimpleVehicle
		float	speed;					// Current speed
		float3	smoothedAcceleration;

		__host__ __device__ float3 velocity( void ) const
		{
			return float3_scalar_multiply( forward, speed );
		}

		__host__ __device__ float3 predictFuturePosition( const float predictionTime ) const
		{
			return float3_add( position, float3_scalar_multiply( velocity(), predictionTime ) );
		}
	} VehicleData;

	typedef struct vehicle_const
	{
		id_type id;

		// SimpleVehicle
		float   maxForce;
		float   maxSpeed;
		float   mass;
		float   radius;
	} VehicleConst;

	//typedef std::vector<VehicleData> DataVec;
	//typedef DataVec::iterator DataVecIt;

	//typedef std::vector<VehicleConst> ConstVec;
	//typedef ConstVec::iterator ConstVecIt;
}//namespace OpenSteer
#endif // VEHICLE_DATA_H