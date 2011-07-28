#include "SteerForSeekCUDA.h"

#include "../VehicleData.cu"

using namespace OpenSteer;

// Kernel function prototype.
extern "C"
{
	__global__ void SteerForSeekCUDAKernel(vehicle_data *vehicleData, float3 target, int numAgents);
}

SteerForSeekCUDA::SteerForSeekCUDA(VehicleGroup *vehicleGroup, const float3 &target)
: AbstractCUDAKernel(vehicleGroup)
{
	m_target = target;
}

void SteerForSeekCUDA::init(void)
{
	// Allocate device memory.
	HANDLE_ERROR(cudaMalloc((void**)&m_pdVehicleData, getDataSizeInBytes()));

	// Copy data to device memory.
	HANDLE_ERROR(cudaMemcpy(m_pdVehicleData, (void*)getVehicleData(), getDataSizeInBytes(), cudaMemcpyHostToDevice));
}

void SteerForSeekCUDA::run(void)
{
	dim3 grid = gridDim();
	dim3 block = blockDim();

	SteerForSeekCUDAKernel<<<grid, block>>>(m_pdVehicleData, m_target, getNumberOfAgents());

	cudaThreadSynchronize();
}

void SteerForSeekCUDA::close(void)
{
	// Copy vehicle data back to the host memory.
	HANDLE_ERROR(cudaMemcpy((void*)getVehicleData(), m_pdVehicleData, getDataSizeInBytes(), cudaMemcpyDeviceToHost));

	// Deallocate device memory
	HANDLE_ERROR(cudaFree(m_pdVehicleData));
	m_pdVehicleData = NULL;
}
