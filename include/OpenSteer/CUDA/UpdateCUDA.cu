#include "UpdateCUDA.h"

using namespace OpenSteer;

// Kernel function prototype.
extern "C"
{
	__global__ void UpdateCUDAKernel(vehicle_data *vehicleData, vehicle_const *vehicleConst, float elapsedTime, int numAgents);
}

UpdateCUDA::UpdateCUDA(VehicleGroup *pVehicleGroup, const float elapsedTime)
:	AbstractCUDAKernel(pVehicleGroup),
	m_elapsedTime(elapsedTime)
{
	m_threadsPerBlock = THREADSPERBLOCK;
}

void UpdateCUDA::init(void)
{
	// Allocate device memory.
	HANDLE_ERROR(cudaMalloc((void**)&m_pdVehicleData, getDataSizeInBytes()));
	HANDLE_ERROR(cudaMalloc((void**)&m_pdVehicleConst, getConstSizeInBytes()));

	// Copy data to device memory.
	HANDLE_ERROR(cudaMemcpy(m_pdVehicleData, (void*)getVehicleData(), getDataSizeInBytes(), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(m_pdVehicleConst, (void*)getVehicleConst(), getConstSizeInBytes(), cudaMemcpyHostToDevice));
}

void UpdateCUDA::run(void)
{
	dim3 grid = gridDim();
	dim3 block = blockDim();

	UpdateCUDAKernel<<<grid, block>>>(m_pdVehicleData, m_pdVehicleConst, m_elapsedTime, getNumberOfAgents());

	cudaThreadSynchronize();
}

void UpdateCUDA::close(void)
{
	// Copy vehicle data back to the host memory.
	HANDLE_ERROR(cudaMemcpy((void*)getVehicleData(), m_pdVehicleData, getDataSizeInBytes(), cudaMemcpyDeviceToHost));

	// Deallocate device memory
	HANDLE_ERROR(cudaFree(m_pdVehicleData));
	HANDLE_ERROR(cudaFree(m_pdVehicleConst));

	m_pdVehicleData = NULL;
	m_pdVehicleConst = NULL;
}
