#include "AvoidObstacleCUDA.h"

using namespace OpenSteer;

// Kernel function prototype.
extern "C"
{
	__global__ void AvoidObstacleCUDAKernel(vehicle_data *vehicleData, vehicle_const *vehicleConst, int numAgents, spherical_obstacle_data *obstacleData, const float minTimeToCollision);
}

AvoidObstacleCUDA::AvoidObstacleCUDA(VehicleGroup *pVehicleGroup, const float minTimeToCollision, const SphericalObstacle *pObstacle)
:	AbstractCUDAKernel(pVehicleGroup),
	m_pObstacle(pObstacle),
	m_minTimeToCollision(minTimeToCollision)
{
}

void AvoidObstacleCUDA::init(void)
{
	// Allocate device memory.
	HANDLE_ERROR(cudaMalloc((void**)&m_pdVehicleData, getDataSizeInBytes()));
	HANDLE_ERROR(cudaMalloc((void**)&m_pdVehicleConst, getConstSizeInBytes()));
	HANDLE_ERROR(cudaMalloc((void**)&m_pdObstacleData, sizeof(spherical_obstacle_data)));

	// Copy data to device memory.
	HANDLE_ERROR(cudaMemcpy(m_pdVehicleData, getVehicleData(), getDataSizeInBytes(), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(m_pdVehicleConst, getVehicleConst(), getConstSizeInBytes(), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(m_pdObstacleData, &m_pObstacle->_data, sizeof(spherical_obstacle_data), cudaMemcpyHostToDevice));
}

void AvoidObstacleCUDA::run(void)
{
	dim3 grid = gridDim();
	dim3 block = blockDim();

	AvoidObstacleCUDAKernel<<<grid, block>>>(m_pdVehicleData, m_pdVehicleConst, getNumberOfAgents(), m_pdObstacleData, m_minTimeToCollision);

	cudaThreadSynchronize();
}

void AvoidObstacleCUDA::close(void)
{
	// Copy vehicle data back to the host memory.
	HANDLE_ERROR(cudaMemcpy(getVehicleData(), m_pdVehicleData, getDataSizeInBytes(), cudaMemcpyDeviceToHost));

	// Deallocate device memory
	HANDLE_ERROR(cudaFree(m_pdVehicleData));
	HANDLE_ERROR(cudaFree(m_pdVehicleConst));
	HANDLE_ERROR(cudaFree(m_pdObstacleData));

	m_pdVehicleData = NULL;
	m_pdVehicleConst = NULL;
	m_pdObstacleData = NULL;
}
