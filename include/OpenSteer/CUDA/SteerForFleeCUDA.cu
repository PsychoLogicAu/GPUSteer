#include "SteerForFleeCUDA.h"

#include "../AgentGroupData.cuh"

using namespace OpenSteer;

// Kernel function prototype.
extern "C"
{
	__global__ void SteerForFleeCUDAKernel(	float3 const*	pdPosition,
											float3 const*	pdForward,
											float3 *		pdSteering,

											float3 const	target,
											size_t const	numAgents,
											float const		fWeight,
											uint *			pdAppliedKernels,
											uint const		doNotApplyWith
											);
}

SteerForFleeCUDA::SteerForFleeCUDA( AgentGroup * pAgentGroup, const float3 &target, float const fWeight, uint const doNotApplyWith )
:	AbstractCUDAKernel( pAgentGroup, fWeight, doNotApplyWith ),
	m_target( target )
{ }

void SteerForFleeCUDA::init(void)
{ }

void SteerForFleeCUDA::run(void)
{
	dim3 grid = gridDim();
	dim3 block = blockDim();

	// Gather required device pointers.
	float3 const*	pdPosition			= m_pAgentGroupData->pdPosition();
	float3 const*	pdForward			= m_pAgentGroupData->pdForward();
	float3 *		pdSteering			= m_pAgentGroupData->pdSteering();

	uint *			pdAppliedKernels	= m_pAgentGroupData->pdAppliedKernels();

	SteerForFleeCUDAKernel<<< grid, block >>>( pdPosition, pdForward, pdSteering, m_target, getNumAgents(), m_fWeight, pdAppliedKernels, m_doNotApplyWith );
	cutilCheckMsg( "SteerForFleeCUDAKernel failed." );
	CUDA_SAFE_CALL( cudaThreadSynchronize() );
}

void SteerForFleeCUDA::close(void)
{
	// Device data has changed. Instruct the AgentGroup it needs to synchronize the host.
	m_pAgentGroup->SetSyncHost();
}
