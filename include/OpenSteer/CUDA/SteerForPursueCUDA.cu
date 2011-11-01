#include "SteerForPursueCUDA.cuh"

#include "../AgentGroupData.cuh"

using namespace OpenSteer;

// Kernel function prototype.
extern "C"
{
__global__ void SteerForPursueCUDAKernel(	float4 const* pdPosition,
											float4 const* pdDirection,
											float const* pdSpeed, 

											float3 const targetPosition,
											float3 const targetForward,
											float3 const targetVelocity,
											float const targetSpeed,

											float4 * pdSteering,

											size_t const numAgents,
											float const maxPredictionTime,
											float const fWeight,
											uint * pdAppliedKernels,
											uint const doNotApplyWith
											);
}

SteerForPursueCUDA::SteerForPursueCUDA(	AgentGroup * pAgentGroup, 
										float3 const& targetPosition,
										float3 const& targetDirection,
										float const& targetSpeed,

										const float fMaxPredictionTime,
										float const fWeight,
										uint const doNotApplyWith
							)
:	AbstractCUDAKernel( pAgentGroup, fWeight, doNotApplyWith ),
	m_targetPosition( targetPosition ),
	m_targetDirection( targetDirection ),
	m_targetSpeed( targetSpeed ),
	m_fMaxPredictionTime( fMaxPredictionTime )
{
	m_targetVelocity = float3_scalar_multiply( m_targetDirection, m_targetSpeed );
}

void SteerForPursueCUDA::init(void)
{ }

void SteerForPursueCUDA::run(void)
{
	dim3 grid = gridDim();
	dim3 block = blockDim();

	// Gether the required device pointers.
	float4 *		pdSteering			= m_pAgentGroupData->pdSteering();
	float4 const*	pdPosition			= m_pAgentGroupData->pdPosition();
	float4 const*	pdDirection			= m_pAgentGroupData->pdDirection();
	float const*	pdSpeed				= m_pAgentGroupData->pdSpeed();
	uint *			pdAppliedKernels	= m_pAgentGroupData->pdAppliedKernels();

	uint const&		numAgents			= getNumAgents();

	SteerForPursueCUDAKernel<<< grid, block >>>(	pdPosition,
													pdDirection,
													pdSpeed,

													m_targetPosition,
													m_targetDirection,
													m_targetVelocity,
													m_targetSpeed,

													pdSteering,

													numAgents,
													m_fMaxPredictionTime,
													m_fWeight,
													pdAppliedKernels,
													m_doNotApplyWith
													);
	cutilCheckMsg( "SteerForPursueCUDAKernel failed." );
	//CUDA_SAFE_CALL( cudaThreadSynchronize() );
}

void SteerForPursueCUDA::close(void)
{
	// Device data has changed. Instruct the AgentGroup it needs to synchronize the host.
	m_pAgentGroup->SetSyncHost();
}
