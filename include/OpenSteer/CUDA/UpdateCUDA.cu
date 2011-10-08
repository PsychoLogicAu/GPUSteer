#include "UpdateCUDA.h"

using namespace OpenSteer;

// Kernel function prototype.
extern "C"
{
	__global__ void UpdateCUDAKernel(	float3 * pdSide,
										float3 * pdUp,
										float4 * pdDirection,
										float4 * pdPosition,

										float4 * pdSteering,
										float * pdSpeed,

										float const* pdMaxForce,
										float const* pdMaxSpeed,
										float const* pdMass,

										float const elapsedTime,
										size_t const numAgents,
										uint * pdAppliedKernels
										);
}

UpdateCUDA::UpdateCUDA( AgentGroup * pAgentGroup, const float fElapsedTime )
:	AbstractCUDAKernel( pAgentGroup, 1.f, 0 ),
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
	float3 *		pdSide				= m_pAgentGroupData->pdSide();
	float3 *		pdUp				= m_pAgentGroupData->pdUp();
	float4 *		pdDirection			= m_pAgentGroupData->pdDirection();
	float4 *		pdPosition			= m_pAgentGroupData->pdPosition();
	float4 *		pdSteering			= m_pAgentGroupData->pdSteering();
	float *			pdSpeed				= m_pAgentGroupData->pdSpeed();

	float const*	pdMaxForce			= m_pAgentGroupData->pdMaxForce();
	float const*	pdMaxSpeed			= m_pAgentGroupData->pdMaxSpeed();
	float const*	pdMass				= m_pAgentGroupData->pdMass();

	uint *			pdAppliedKernels	= m_pAgentGroupData->pdAppliedKernels();

	uint const&		numAgents			= getNumAgents();

	UpdateCUDAKernel<<< grid, block >>>(	pdSide, pdUp, pdDirection, pdPosition, pdSteering, pdSpeed,
											pdMaxForce, pdMaxSpeed, pdMass,
											m_fElapsedTime, numAgents,
											pdAppliedKernels
											);
	cutilCheckMsg( "UpdateCUDAKernel failed." );
	CUDA_SAFE_CALL( cudaThreadSynchronize() );
}

void UpdateCUDA::close(void)
{
	// Device data has changed. Instruct the AgentGroup it needs to synchronize the host.
	m_pAgentGroup->SetSyncHost();
}
