#include "SteerForPursuitCUDA.h"

#include "../VehicleData.cu"

using namespace OpenSteer;

// Kernel function prototype.
extern "C"
{
	__global__ void SteerForPursuitCUDAKernel(vehicle_data *vehicleData, vehicle_data *target, const int numAgents, const float maxPredictionTime);
}

SteerForPursuitCUDA::SteerForPursuitCUDA(VehicleGroup *pVehicleGroup, const vehicle_data *pTarget, const float maxPredictionTime)
:	AbstractCUDAKernel(pVehicleGroup),
	m_pTarget(pTarget),
	m_maxPredictionTime(maxPredictionTime)
{
}

void SteerForPursuitCUDA::init(void)
{
	size_t av_size = sizeof(vehicle_data);
	size_t vdata_size = getDataSizeInBytes();

	// Allocate device memory.
	HANDLE_ERROR(cudaMalloc((void**)&m_pdVehicleData, vdata_size));
	HANDLE_ERROR(cudaMalloc((void**)&m_pdTarget, av_size));

	// Copy data to device memory.
	HANDLE_ERROR(cudaMemcpy(m_pdVehicleData, getVehicleData(), vdata_size, cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(m_pdTarget, m_pTarget, av_size, cudaMemcpyHostToDevice));
}

void SteerForPursuitCUDA::run(void)
{
	dim3 grid = gridDim();
	dim3 block = blockDim();

	SteerForPursuitCUDAKernel<<<grid, block>>>(m_pdVehicleData, m_pdTarget, getNumberOfAgents(), m_maxPredictionTime);

	cudaThreadSynchronize();
}

void SteerForPursuitCUDA::close(void)
{
	// Copy vehicle data back to the host memory.
	HANDLE_ERROR(cudaMemcpy((void*)getVehicleData(), m_pdVehicleData, getDataSizeInBytes(), cudaMemcpyDeviceToHost));

	// Deallocate device memory
	HANDLE_ERROR(cudaFree(m_pdVehicleData));
	HANDLE_ERROR(cudaFree(m_pdTarget));
	
	m_pdVehicleData = NULL;
	m_pdTarget = NULL;
}
