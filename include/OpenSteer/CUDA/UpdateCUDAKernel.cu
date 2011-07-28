#include "UpdateCUDA.h"

#include "../VehicleData.cu"
#include "../VectorUtils.cu"

#include "CUDAKernelGlobals.h"

using namespace OpenSteer;

extern "C"
{
	__global__ void UpdateCUDAKernel(vehicle_data *vehicleData, vehicle_const *vehicleConst, float elapsedTime, int numAgents)
	{
		int offset = (blockIdx.x * blockDim.x) + threadIdx.x;

		// Check bounds.
		if(offset >= numAgents)
			return;

		// Copy the vehicleData and vehicleConst values to shared memory.
		__shared__ vehicle_data vehicleDataShared[THREADSPERBLOCK];
		__shared__ vehicle_const vehicleConstShared[THREADSPERBLOCK];

		VDATA_SH(threadIdx.x) = VDATA(offset);
		VCONST_SH(threadIdx.x) = VCONST(offset);

		__syncthreads();

		// If we don't have a steering vector set, do nothing.
		if(float3_equals(STEERING_SH(threadIdx.x), float3_zero()))
			return;

		// Enforce limit on magnitude of steering force.
		STEERING_SH(threadIdx.x) = float3_truncateLength(STEERING_SH(threadIdx.x), MAXFORCE_SH(threadIdx.x));

		// Compute acceleration and velocity.
		float3 newAcceleration = float3_scalar_divide(STEERING_SH(threadIdx.x), MASS_SH(threadIdx.x));
		float3 newVelocity = float3_add(VELOCITY_SH(threadIdx.x), float3_scalar_multiply(newAcceleration, elapsedTime));

		// Enforce speed limit.
		newVelocity = float3_truncateLength(newVelocity, MAXSPEED_SH(threadIdx.x));

		// Update speed.
		SPEED_SH(threadIdx.x) = float3_length(newVelocity);

		if(SPEED_SH(threadIdx.x) > 0)
		{
			// Calculate the unit forward vector.
			FORWARD_SH(threadIdx.x) = float3_scalar_divide(newVelocity, SPEED_SH(threadIdx.x));

			// derive new side basis vector from NEW forward and OLD up.
			SIDE_SH(threadIdx.x) = float3_normalize(float3_cross(FORWARD_SH(threadIdx.x), UP_SH(threadIdx.x))); // TODO: handedness? assumed right

			// derive new up basis vector from new forward and side.
			UP_SH(threadIdx.x) = float3_cross(SIDE_SH(threadIdx.x), FORWARD_SH(threadIdx.x)); // TODO: handedness? assumed right
		}

		// Euler integrate (per frame) velocity into position.
		POSITION_SH(threadIdx.x) = float3_add(POSITION_SH(threadIdx.x), float3_scalar_multiply(newVelocity, elapsedTime));

		// Set the steering vector back to zero.
		STEERING_SH(threadIdx.x) = float3_zero();

		__syncthreads();

		// Copy the data back into global memory.
		VDATA(offset) = VDATA_SH(threadIdx.x);
	}
}
