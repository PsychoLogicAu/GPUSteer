#include "SteerForFleeCUDA.h"

#include "../VehicleGroupData.cu"

using namespace OpenSteer;

// Kernel function prototype.
extern "C"
{
	__global__ void SteerForFleeCUDAKernel(	float3 const* pdPosition, float3 const* pdForward, float3 * pdSteering,
											float3 const target, size_t const numAgents );
}

SteerForFleeCUDA::SteerForFleeCUDA( VehicleGroup * pVehicleGroup, const float3 &target )
:	AbstractCUDAKernel( pVehicleGroup ),
	m_target( target )
{ }

void SteerForFleeCUDA::init(void)
{ }

void SteerForFleeCUDA::run(void)
{
	dim3 grid = gridDim();
	dim3 block = blockDim();

	// Gather required device pointers.
	float3 const* pdPosition = m_pdVehicleGroupData->dpPosition();
	float3 const* pdForward = m_pdVehicleGroupData->dpForward();
	float3 * pdSteering = m_pdVehicleGroupData->dpSteering();

	SteerForFleeCUDAKernel<<< grid, block >>>( pdPosition, pdForward, pdSteering, m_target, getNumAgents() );

	cudaThreadSynchronize();
}

void SteerForFleeCUDA::close(void)
{
	// Device data has changed. Instruct the VehicleGroup it needs to synchronize the host.
	m_pVehicleGroup->SetSyncHost();
}
