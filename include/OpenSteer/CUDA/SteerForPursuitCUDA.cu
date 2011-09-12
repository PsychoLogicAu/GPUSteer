#include "SteerForPursuitCUDA.h"

#include "../VehicleGroupData.cuh"

using namespace OpenSteer;

// Kernel function prototype.
extern "C"
{
	__global__ void SteerForPursuitCUDAKernel(	float3 * pdSteering, float3 const* pdPosition, float3 const* pdForward, float const* pdSpeed, 
												float3 const targetPosition, float3 const targetForward, float3 const targetVelocity, float const targetSpeed,
												size_t const numAgents, float const maxPredictionTime, float const weight
												);
}

SteerForPursuitCUDA::SteerForPursuitCUDA(	VehicleGroup * pVehicleGroup, 
											float3 const& targetPosition, float3 const& targetForward, float3 const& targetVelocity, float const& targetSpeed,
											const float fMaxPredictionTime, float const fWeight )
:	AbstractCUDAKernel( pVehicleGroup, fWeight ),
	m_targetPosition( targetPosition ),
	m_targetForward( targetForward ),
	m_targetVelocity( targetVelocity ),
	m_targetSpeed( targetSpeed ),
	m_fMaxPredictionTime( fMaxPredictionTime )
{ }

void SteerForPursuitCUDA::init(void)
{ }

void SteerForPursuitCUDA::run(void)
{
	dim3 grid = gridDim();
	dim3 block = blockDim();

	// Gether the required device pointers.
	float3 * pdSteering = m_pVehicleGroupData->pdSteering();
	float3 const* pdPosition = m_pVehicleGroupData->pdPosition();
	float3 const* pdForward = m_pVehicleGroupData->pdForward();
	float const* pdSpeed = m_pVehicleGroupData->pdSpeed();

	SteerForPursuitCUDAKernel<<< grid, block >>>( pdSteering, pdPosition, pdForward, pdSpeed, m_targetPosition, m_targetForward, m_targetVelocity, m_targetSpeed, getNumAgents(), m_fMaxPredictionTime, m_fWeight );
	cutilCheckMsg( "SteerForPursuitCUDAKernel failed." );
	CUDA_SAFE_CALL( cudaThreadSynchronize() );
}

void SteerForPursuitCUDA::close(void)
{
	// Device data has changed. Instruct the VehicleGroup it needs to synchronize the host.
	m_pVehicleGroup->SetSyncHost();
}
