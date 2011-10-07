#include "SteerForSeekCUDA.cuh"

#include "../AgentGroupData.cuh"

using namespace OpenSteer;

// Kernel function prototype.
extern "C"
{
	__global__ void SteerForSeekCUDAKernel(	float3 *		pdSteering,
											float3 const*	pdPosition,
											float3 const*	pdForward,
											float3 const	target,
											size_t const	numAgents,
											float const		fWeight,
											uint *			pdAppliedKernels,
											uint const		doNotApplyWith
											);
}

SteerForSeekCUDA::SteerForSeekCUDA( AgentGroup * pAgentGroup, float3 const& target, float const fWeight, uint const doNotApplyWith )
:	AbstractCUDAKernel( pAgentGroup, fWeight, doNotApplyWith ),
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
	float3 *		pdSteering			= m_pAgentGroupData->pdSteering();
	float3 const*	pdPosition			= m_pAgentGroupData->pdPosition();
	float3 const*	pdForward			= m_pAgentGroupData->pdForward();

	uint *			pdAppliedKernels	= m_pAgentGroupData->pdAppliedKernels();

	SteerForSeekCUDAKernel<<< grid, block >>>( pdSteering, pdPosition, pdForward, m_target, getNumAgents(), m_fWeight, pdAppliedKernels, m_doNotApplyWith );
	cutilCheckMsg( "SteerForSeekCUDAKernel failed." );
	CUDA_SAFE_CALL( cudaThreadSynchronize() );
}

void SteerForSeekCUDA::close(void)
{
	// Device data has changed. Instruct the AgentGroup it needs to synchronize the host.
	m_pAgentGroup->SetSyncHost();
}
