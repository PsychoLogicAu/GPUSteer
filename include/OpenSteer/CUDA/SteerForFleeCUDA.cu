#include "SteerForFleeCUDA.cuh"

#include "../AgentGroupData.cuh"

using namespace OpenSteer;

// Kernel function prototype.
extern "C"
{
	__global__ void SteerForFleeCUDAKernel(	float4 const*	pdPosition,
											float4 const*	pdDirection,
											float4 *		pdSteering,

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
	float4 const*	pdPosition			= m_pAgentGroupData->pdPosition();
	float4 const*	pdDirection			= m_pAgentGroupData->pdDirection();
	float4 *		pdSteering			= m_pAgentGroupData->pdSteering();

	uint *			pdAppliedKernels	= m_pAgentGroupData->pdAppliedKernels();

	SteerForFleeCUDAKernel<<< grid, block >>>( pdPosition, pdDirection, pdSteering, m_target, getNumAgents(), m_fWeight, pdAppliedKernels, m_doNotApplyWith );
	cutilCheckMsg( "SteerForFleeCUDAKernel failed." );
	//CUDA_SAFE_CALL( cudaThreadSynchronize() );
}

void SteerForFleeCUDA::close(void)
{
	// Device data has changed. Instruct the AgentGroup it needs to synchronize the host.
	m_pAgentGroup->SetSyncHost();
}
