#include "AvoidObstaclesCUDA.h"

using namespace OpenSteer;

// Kernel function prototype.
extern "C"
{
	__global__ void AvoidObstaclesCUDAKernel(vehicle_data *vehicleData, vehicle_const *vehicleConst, int numAgents, spherical_obstacle_data *obstacleData, near_obstacle_index *nearObstacleIndices, int *obstacleIndices, const float minTimeToCollision);
}

AvoidObstaclesCUDA::AvoidObstaclesCUDA(VehicleGroup *pVehicleGroup, const float minTimeToCollision, ObstacleGroup *pObstacleGroup)
:	AbstractCUDAKernel(pVehicleGroup),
	m_pObstacleGroup(pObstacleGroup),
	m_minTimeToCollision(minTimeToCollision)
{
}

void AvoidObstaclesCUDA::init(void)
{
	m_runKernel = true;

	// Indices into the m_pObstacleGroup data array.
	std::vector<int> obstacleIndices;

	int numVehicles = m_pVehicleGroup->Size();

	// Indices for each vehicle into the obstacleIndices vector.
	//NearObstacleIndex *pVehicleIndices = new NearObstacleIndex[numVehicles];
	std::vector<NearObstacleIndex> vehicleIndices(numVehicles);

	const VehicleData* vehicleData = m_pVehicleGroup->GetVehicleData();
	const VehicleConst* vehicleConst = m_pVehicleGroup->GetVehicleConst();

	// For each vehicle in the group.
	for(int i = 0; i < numVehicles; i++)
	{
		const VehicleData &vData = vehicleData[i];
		const VehicleConst &vConst = vehicleConst[i];

		SphericalObstacleDataVec nearObstacles;

		// Get the indices of the obstacles which are near the vehicle.
		m_pObstacleGroup->FindNearObstacles(vData.position, /*vConst.radius * 60.0f*/ 8.0f, nearObstacles);

		if(nearObstacles.size() == 0) // No near obstacles for this vehicle.
		{
			// Set to -1 to indicate there is no near obstacle.
			vehicleIndices[i].numObstacles = 0;
			continue;
		}

		// The first obstacle index for this vehicle will be the next available slot in obstacleIndices
		vehicleIndices[i].numObstacles = nearObstacles.size();
		vehicleIndices[i].baseIndex = obstacleIndices.size();

		for(unsigned int j = 0; j < nearObstacles.size(); j++)
		{
			obstacleIndices.push_back(nearObstacles[j]->id);
		}
	}

	if(obstacleIndices.size() == 0) // There were no obstacles detected within a vehicle's range.  Running the kernel is unnecesary and will cause a crash.
	{
		m_runKernel = false;
		return;
	}

	std::vector<SphericalObstacleData>& obstacleData = m_pObstacleGroup->m_vObstacleData;

	size_t vehicleIndicesSize	= numVehicles * sizeof(NearObstacleIndex);
	size_t obstacleIndicesSize	= obstacleIndices.size() * sizeof(int);
	size_t obstacleDataSize		= obstacleData.size() * sizeof(SphericalObstacleData);

	// Allocate device memory.
	HANDLE_ERROR(cudaMalloc((void**)&m_pdVehicleData, getDataSizeInBytes()));
	HANDLE_ERROR(cudaMalloc((void**)&m_pdVehicleConst, getConstSizeInBytes()));
	HANDLE_ERROR(cudaMalloc((void**)&m_pdNearObstacleIndices, vehicleIndicesSize));
	HANDLE_ERROR(cudaMalloc((void**)&m_pdObstacleIndices, obstacleIndicesSize));
	HANDLE_ERROR(cudaMalloc((void**)&m_pdObstacleData, obstacleDataSize));

	// Copy data to device memory.
	HANDLE_ERROR(cudaMemcpy(m_pdVehicleData, getVehicleData(), getDataSizeInBytes(), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(m_pdVehicleConst, getVehicleConst(), getConstSizeInBytes(), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(m_pdNearObstacleIndices, &vehicleIndices[0], vehicleIndicesSize, cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(m_pdObstacleIndices, &obstacleIndices[0], obstacleIndicesSize, cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(m_pdObstacleData, &obstacleData[0], obstacleDataSize, cudaMemcpyHostToDevice));
}

void AvoidObstaclesCUDA::run(void)
{
	if(m_runKernel)
	{
		dim3 grid = gridDim();
		dim3 block = blockDim();

		AvoidObstaclesCUDAKernel<<<grid, block>>>(m_pdVehicleData, m_pdVehicleConst, getNumberOfAgents(), m_pdObstacleData, m_pdNearObstacleIndices, m_pdObstacleIndices, m_minTimeToCollision);

		cudaThreadSynchronize();
	}
}

void AvoidObstaclesCUDA::close(void)
{
	if(m_runKernel)
	{
		// Copy vehicle data back to the host memory.
		HANDLE_ERROR(cudaMemcpy(getVehicleData(), m_pdVehicleData, getDataSizeInBytes(), cudaMemcpyDeviceToHost));

		// Deallocate device memory
		HANDLE_ERROR(cudaFree(m_pdVehicleData));
		HANDLE_ERROR(cudaFree(m_pdVehicleConst));
		HANDLE_ERROR(cudaFree(m_pdNearObstacleIndices));
		HANDLE_ERROR(cudaFree(m_pdObstacleIndices));
		HANDLE_ERROR(cudaFree(m_pdObstacleData));

		m_pdVehicleData = NULL;
		m_pdVehicleConst = NULL;
		m_pdNearObstacleIndices = NULL;
		m_pdObstacleIndices = NULL;
		m_pdObstacleData = NULL;
	}
}
