#include "WrapWorldCUDA.cuh"

extern "C"
{
	__global__ void WrapWorldKernel(	float4 *	pdPosition,
										float3 const	worldSize,
										uint const numAgents
										);
}

using namespace OpenSteer;

WrapWorldCUDA::WrapWorldCUDA( AgentGroup * pAgentGroup, float3 const& worldSize )
:	AbstractCUDAKernel( pAgentGroup, 0.f, 0 ),
	m_worldSize( worldSize )
{
	// Nothing to do.
}

void WrapWorldCUDA::init( void )
{
	// Nothing to do.
}

void WrapWorldCUDA::run( void )
{
	dim3 grid = gridDim();
	dim3 block = blockDim();

	float4 *		pdPosition	= m_pAgentGroupData->pdPosition();
	uint const		numAgents	= m_pAgentGroupData->size();

	WrapWorldKernel<<< grid, block >>>(	pdPosition,
										m_worldSize,
										numAgents
										);
	cutilCheckMsg( "WrapWorldKernel failed." );
	CUDA_SAFE_CALL( cudaThreadSynchronize() );
}

void WrapWorldCUDA::close( void )
{
	// Agent data has possibly changed.
	m_pAgentGroup->SetSyncHost();
}
