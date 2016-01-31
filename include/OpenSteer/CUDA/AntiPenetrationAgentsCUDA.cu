#include "AntiPenetrationAgentsCUDA.cuh"

extern "C"
{
	__host__ void AntiPenetrationAgentsKernelBindTextures(	float4 const*	pdBPosition,
															float const*	pdBRadius,
															uint const*		pdBAppliedKernels,
															uint const		numB
															);

	__host__ void AntiPenetrationAgentsKernelUnbindTextures( void );

	__global__ void AntiPenetrationAgentsCUDAKernel(		float4 const*	pdPosition,
															float const*	pdRadius,
															uint const		numA,

															uint const*		pdKNNIndices,
															float const*	pdKNNDistances,
															uint const		k,

															uint const		numB,

															float4 *		pdPositionOut,
															
															uint *			pdAppliedKernels,
															uint const		doNotApplyWith
															);
}

using namespace OpenSteer;

AntiPenetrationAgentsCUDA::AntiPenetrationAgentsCUDA( AgentGroup * pAgentGroup, KNNData * pKNNData, AgentGroup * pOtherGroup, uint const doNotApplyWith )
:	AbstractCUDAKernel( pAgentGroup, 0.f, doNotApplyWith ),
	m_pKNNData( pKNNData ),
	m_pOtherGroup( pOtherGroup )
{
	// Nothing to do.
}

void AntiPenetrationAgentsCUDA::init( void )
{
	// Allocate temporary storage for the new positions.
	CUDA_SAFE_CALL( cudaMalloc( &m_pdPositionNew, getNumAgents() * sizeof(float4) ) );
}

void AntiPenetrationAgentsCUDA::run( void )
{
	dim3 grid = gridDim();
	dim3 block = blockDim();

	// Gather required device pointers.
	float4 *		pdPosition			= m_pAgentGroupData->pdPosition();
	float const*	pdRadius			= m_pAgentGroupData->pdRadius();
	uint const&		numA				= getNumAgents();

	uint const*		pdKNNIndices		= m_pKNNData->pdKNNIndices();
	float const*	pdKNNDistances		= m_pKNNData->pdKNNDistances();
	uint const&		k					= m_pKNNData->k();

	float4 const*	pdBPosition			= m_pOtherGroup->pdPosition();
	float const*	pdBRadius			= m_pOtherGroup->pdRadius();
	uint const&		numB				= m_pOtherGroup->Size();

	uint *			pdAppliedKernels	= m_pAgentGroupData->pdAppliedKernels();
	uint const*		pdBAppliedKernels	= m_pOtherGroup->GetAgentGroupData().pdAppliedKernels();

	// Bind the textures.
	AntiPenetrationAgentsKernelBindTextures( pdBPosition, pdBRadius, pdBAppliedKernels, numB );

	// Copy the current positions into m_pdPositionNew.
	CUDA_SAFE_CALL( cudaMemcpy( m_pdPositionNew, pdPosition, numA * sizeof(float4), cudaMemcpyDeviceToDevice ) );

	AntiPenetrationAgentsCUDAKernel<<< grid, block >>>( pdPosition,
														pdRadius,
														numA,

														pdKNNIndices,
														pdKNNDistances,
														k,

														numB,

														m_pdPositionNew,

														pdAppliedKernels,
														m_doNotApplyWith
														);
	cutilCheckMsg( "AntiPenetrationAgentsCUDAKernel failed." );
	//CUDA_SAFE_CALL( cudaThreadSynchronize() );

	// Unbind the textures.
	AntiPenetrationAgentsKernelUnbindTextures();

	// Copy the new position data back to the AgentGroup.
	CUDA_SAFE_CALL( cudaMemcpy( pdPosition, m_pdPositionNew, numA * sizeof(float4), cudaMemcpyDeviceToDevice ) );
}

void AntiPenetrationAgentsCUDA::close( void )
{
	// Free the temporary storage.
	CUDA_SAFE_CALL( cudaFree( m_pdPositionNew ) );
}
