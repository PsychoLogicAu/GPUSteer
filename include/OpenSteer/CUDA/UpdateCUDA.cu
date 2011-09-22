#include "UpdateCUDA.h"

using namespace OpenSteer;

// Kernel function prototype.
extern "C"
{
	__global__ void UpdateCUDAKernel(	// vehicle_group_data members.
										float3 * pdSide, float3 * pdUp, float3 * pdDirection,
										float3 * pdPosition, float3 * pdSteering, float * pdSpeed,
										// vehicle_group_const members.
										float const* pdMaxForce, float const* pdMaxSpeed, float const* pdMass,
										float const elapsedTime, size_t const numAgents,
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
	float3 *		pdForward			= m_pAgentGroupData->pdForward();
	float3 *		pdPosition			= m_pAgentGroupData->pdPosition();
	float3 *		pdSteering			= m_pAgentGroupData->pdSteering();
	float *			pdSpeed				= m_pAgentGroupData->pdSpeed();

	float const*	pdMaxForce			= m_pAgentGroupConst->pdMaxForce();
	float const*	pdMaxSpeed			= m_pAgentGroupConst->pdMaxSpeed();
	float const*	pdMass				= m_pAgentGroupConst->pdMass();

	uint *			pdAppliedKernels	= m_pAgentGroupData->pdAppliedKernels();

	UpdateCUDAKernel<<< grid, block >>>(	pdSide, pdUp, pdForward, pdPosition, pdSteering, pdSpeed,
											pdMaxForce, pdMaxSpeed, pdMass,
											m_fElapsedTime, getNumAgents(),
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
