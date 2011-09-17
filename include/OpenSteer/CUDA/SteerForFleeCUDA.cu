#include "SteerForFleeCUDA.h"

#include "../AgentGroupData.cuh"

using namespace OpenSteer;

// Kernel function prototype.
extern "C"
{
	__global__ void SteerForFleeCUDAKernel(	float3 const* pdPosition, float3 const* pdForward, float3 * pdSteering,
											float3 const target, size_t const numAgents, float const fWeight );
}

SteerForFleeCUDA::SteerForFleeCUDA( AgentGroup * pAgentGroup, const float3 &target, float const fWeight )
:	AbstractCUDAKernel( pAgentGroup, fWeight ),
	m_target( target )
{ }

void SteerForFleeCUDA::init(void)
{ }

void SteerForFleeCUDA::run(void)
{
	dim3 grid = gridDim();
	dim3 block = blockDim();

	// Gather required device pointers.
	float3 const* pdPosition = m_pAgentGroupData->pdPosition();
	float3 const* pdForward = m_pAgentGroupData->pdForward();
	float3 * pdSteering = m_pAgentGroupData->pdSteering();

	SteerForFleeCUDAKernel<<< grid, block >>>( pdPosition, pdForward, pdSteering, m_target, getNumAgents(), m_fWeight );
	cutilCheckMsg( "SteerForFleeCUDAKernel failed." );

	cudaThreadSynchronize();
}

void SteerForFleeCUDA::close(void)
{
	// Device data has changed. Instruct the AgentGroup it needs to synchronize the host.
	m_pAgentGroup->SetSyncHost();
}
