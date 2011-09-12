#include "SteerForSeekCUDA.h"

#include "../VehicleGroupData.cuh"

using namespace OpenSteer;

// Kernel function prototype.
extern "C"
{
	__global__ void SteerForSeekCUDAKernel( float3 * pdSteering, float3 const* pdPosition, float3 const* pdForward, float3 const target, size_t const numAgents, float const fWeight );
}

SteerForSeekCUDA::SteerForSeekCUDA( VehicleGroup * pVehicleGroup, float3 const& target, float const fWeight )
:	AbstractCUDAKernel( pVehicleGroup, fWeight ),
	m_target( target )
{
}

void SteerForSeekCUDA::init( void )
{
	// Nothing to do.
}

void SteerForSeekCUDA::run(void)
{
	dim3 grid = gridDim();
	dim3 block = blockDim();

	// Gather the required device pointers.
	float3 * pdSteering = m_pVehicleGroupData->pdSteering();
	float3 const* pdPosition = m_pVehicleGroupData->pdPosition();
	float3 const* pdForward = m_pVehicleGroupData->pdForward();

	SteerForSeekCUDAKernel<<< grid, block >>>( pdSteering, pdPosition, pdForward, m_target, getNumAgents(), m_fWeight );
	cutilCheckMsg( "SteerForSeekCUDAKernel failed." );
	CUDA_SAFE_CALL( cudaThreadSynchronize() );
}

void SteerForSeekCUDA::close(void)
{
	// Device data has changed. Instruct the VehicleGroup it needs to synchronize the host.
	m_pVehicleGroup->SetSyncHost();
}
