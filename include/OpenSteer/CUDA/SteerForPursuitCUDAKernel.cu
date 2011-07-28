#include "SteerForPursuitCUDA.h"

#include "../VehicleData.cu"
#include "../VectorUtils.cu"

#include "CUDAKernelGlobals.h"

using namespace OpenSteer;

extern "C"
{
	__global__ void SteerForPursuitCUDAKernel(vehicle_data *vehicleData, vehicle_data *target, const int numAgents, const float maxPredictionTime)
	{
		int offset = (blockIdx.x * blockDim.x) + threadIdx.x;

		// Check bounds.
		if(offset >= numAgents)
			return;

		// If we already have a steering vector set, do nothing.
		if(!float3_equals(STEERING(offset), float3_zero()))
			return;

		// If the target is ahead, just seek to its current position.
		float3 toTarget = float3_subtract(target->position, POSITION(offset));
		float relativeHeading = float3_dot(FORWARD(offset), target->forward);

		if(float3_dot(toTarget, FORWARD(offset)) > 0 && (relativeHeading < -0.95f))
		{
			// Get the desired velocity.
			float3 desiredVelocity = float3_subtract(target->position, POSITION(offset));

			// Set the steering vector.
			STEERING(offset) = float3_subtract(desiredVelocity, FORWARD(offset));
			return;
		}

		float lookAheadTime = float3_length(toTarget) / (SPEED(offset) + target->speed);
		float3 newTarget = float3_add(target->position, float3_scalar_multiply(target->velocity(), (maxPredictionTime < lookAheadTime) ? maxPredictionTime : lookAheadTime));

		// Get the desired velocity.
		float3 desiredVelocity = float3_subtract(newTarget, POSITION(offset));

		// Set the steering vector.
		STEERING(offset) = float3_subtract(desiredVelocity, FORWARD(offset));
	}
}