#include "SteerForSeekCUDA.cuh"

#include "../AgentGroupData.cuh"

using namespace OpenSteer;

// Kernel function prototype.
extern "C"
{
	__global__ void SteerForSeekCUDAKernel(	float4 *		pdSteering,
											float4 const*	pdPosition,
											float4 const*	pdDirection,

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
	float4 *		pdSteering			= m_pAgentGroupData->pdSteering();
	float4 const*	pdPosition			= m_pAgentGroupData->pdPosition();
	float4 const*	pdDirection			= m_pAgentGroupData->pdDirection();

	uint *			pdAppliedKernels	= m_pAgentGroupData->pdAppliedKernels();

	uint const&		numAgents			= getNumAgents();

	SteerForSeekCUDAKernel<<< grid, block >>>( pdSteering,
												pdPosition,
												pdDirection,
												
												m_target,
												
												numAgents,
												m_fWeight,
												pdAppliedKernels,
												m_doNotApplyWith
												);
	cutilCheckMsg( "SteerForSeekCUDAKernel failed." );
	//CUDA_SAFE_CALL( cudaThreadSynchronize() );
}

void SteerForSeekCUDA::close(void)
{
	// Device data has changed. Instruct the AgentGroup it needs to synchronize the host.
	m_pAgentGroup->SetSyncHost();
}
