#include "SteerForSeekCUDA.h"

#include "../VehicleData.cu"
#include "../VectorUtils.cu"

#include "CUDAKernelGlobals.h"

using namespace OpenSteer;

extern "C"
{
	__global__ void SteerForSeekCUDAKernel(VehicleData *vehicleData, float3 target, int numAgents)
	{
		int offset = (blockIdx.x * blockDim.x) + threadIdx.x;

		// Check bounds.
		if(offset >= numAgents)
			return;

		// If we already have a steering vector set, do nothing.
		if(!float3_equals(STEERING(offset), float3_zero()))
			return;

		__shared__ float3 desiredVelocity[THREADSPERBLOCK];

		// Get the desired velocity.
		desiredVelocity[threadIdx.x] = float3_subtract(target, POSITION(offset));

		// Set the steering vector.
		STEERING(offset) = float3_subtract(desiredVelocity[threadIdx.x], FORWARD(offset));
	}

	//__global__ void SteerForSeekKernel(vehicle_data *vehicleData, float3 *seekVectors, int numAgents)
	//{
	//	int offset = (blockIdx.x * blockDim.x) + threadIdx.x;

	//	// Stay within group bounds.
	//	if(offset > numAgents)
	//		return;

	//	// If we already have a steering vector set, do nothing.
	//	if(!float3_equals(STEERING(offset), float3_zero()))
	//		return;

	//	const float3 desiredVelocity = float3_subtract(SEEK(offset), VEHICLE(offset).position);
	//	STEERING(offset) = float3_subtract(desiredVelocity, FORWARD(offset));
	//}

}