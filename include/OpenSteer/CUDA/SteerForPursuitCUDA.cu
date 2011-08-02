#include "SteerForPursuitCUDA.h"

#include "../VehicleGroupData.h"

using namespace OpenSteer;

// Kernel function prototype.
extern "C"
{
	__global__ void SteerForPursuitCUDAKernel(	float3 * pdSteering, float3 const* pdPosition, float3 const* pdForward, float const* pdSpeed, 
												float3 const targetPosition, float3 const targetForward, float3 const targetVelocity, float const targetSpeed,
												size_t const numAgents, float const maxPredictionTime );
}

SteerForPursuitCUDA::SteerForPursuitCUDA(	VehicleGroup * pVehicleGroup, 
											float3 const& targetPosition, float3 const& targetForward, float3 const& targetVelocity, float const& targetSpeed,
											const float fMaxPredictionTime )
:	AbstractCUDAKernel( pVehicleGroup ),
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
	float3 * pdSteering = m_pdVehicleGroupData->dpSteering();
	float3 const* pdPosition = m_pdVehicleGroupData->dpPosition();
	float3 const* pdForward = m_pdVehicleGroupData->dpForward();
	float const* pdSpeed = m_pdVehicleGroupData->dpSpeed();

	SteerForPursuitCUDAKernel<<< grid, block >>>( pdSteering, pdPosition, pdForward, pdSpeed, m_targetPosition, m_targetForward, m_targetVelocity, m_targetSpeed, getNumAgents(), m_fMaxPredictionTime );

	cudaThreadSynchronize();
}

void SteerForPursuitCUDA::close(void)
{
	// Device data has changed. Instruct the VehicleGroup it needs to synchronize the host.
	m_pVehicleGroup->SetSyncHost();
}
