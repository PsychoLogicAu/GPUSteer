#ifndef OPENSTEER_VEHICLEDATA_H
#define OPENSTEER_VEHICLEDATA_H

#include <vector>
#include <cuda_runtime.h>

#include "VectorUtils.cuh"

namespace OpenSteer
{
	typedef unsigned int id_type;

	typedef struct agent_data
	{
		id_type	id;

		float3	side;					// Side vector
		float3	up;						// Up vector
		float4	direction;
		float4	position;

		float4	steering;
		float	speed;					// Current speed

		float	maxSpeed;
		float	maxForce;
		float	radius;
		float	mass;

		uint	appliedKernels;
		
		__host__ __device__ float3 velocity( void ) const
		{
			return float3_scalar_multiply( make_float3( direction ), speed );
		}

		__host__ __device__ float3 predictFuturePosition( const float predictionTime ) const
		{
			return float3_add( make_float3( position ), float3_scalar_multiply( velocity(), predictionTime ) );
		}

	} AgentData;

}//namespace OpenSteer
#endif // VEHICLE_DATA_H