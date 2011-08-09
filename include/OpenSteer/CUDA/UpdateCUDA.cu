#include "UpdateCUDA.h"

using namespace OpenSteer;

// Kernel function prototype.
extern "C"
{
	__global__ void UpdateCUDAKernel(	// vehicle_group_data members.
										float3 * pdSide, float3 * pdUp, float3 * pdForward,
										float3 * pdPosition, float3 * pdSteering, float * pdSpeed,
										// vehicle_group_const members.
										float const* pdMaxForce, float const* pdMaxSpeed, float const* pdMass,
										float const elapsedTime, size_t const numAgents );
}

UpdateCUDA::UpdateCUDA( VehicleGroup * pVehicleGroup, const float fElapsedTime )
:	AbstractCUDAKernel( pVehicleGroup ),
	m_fElapsedTime( fElapsedTime )
{
}

void UpdateCUDA::init( void )
{
	// Nothing to do.
}

void UpdateCUDA::run(void)
{
	dim3 grid = gridDim();
	dim3 block = blockDim();

	// Gather pointers to the required data...
	float3 * pdSide = m_pVehicleGroupData->pdSide();
	float3 * pdUp = m_pVehicleGroupData->pdUp();
	float3 * pdForward = m_pVehicleGroupData->pdForward();
	float3 * pdPosition = m_pVehicleGroupData->pdPosition();
	float3 * pdSteering = m_pVehicleGroupData->pdSteering();
	float * pdSpeed = m_pVehicleGroupData->pdSpeed();

	float const* pdMaxForce = m_pVehicleGroupConst->pdMaxForce();
	float const* pdMaxSpeed = m_pVehicleGroupConst->pdMaxSpeed();
	float const* pdMass = m_pVehicleGroupConst->pdMass();

	UpdateCUDAKernel<<< grid, block >>>(	pdSide, pdUp, pdForward, pdPosition, pdSteering, pdSpeed,
											pdMaxForce, pdMaxSpeed, pdMass,
											m_fElapsedTime, getNumAgents() );

	cudaThreadSynchronize();
}

void UpdateCUDA::close(void)
{
	// Device data has changed. Instruct the VehicleGroup it needs to synchronize the host.
	m_pVehicleGroup->SetSyncHost();
}
