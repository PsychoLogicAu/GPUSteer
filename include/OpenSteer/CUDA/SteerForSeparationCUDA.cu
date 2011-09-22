#include "SteerForSeparationCUDA.cuh"

using namespace OpenSteer;

extern "C"
{
	__global__ void SteerForSeparationKernel(	uint const*		pdKNNIndices,
												size_t const	k,
												
												float3 const*	pdPosition,
		
												float3 *		pdSteering,
												size_t const	numAgents,
												float const		fWeight,

												uint *			pdAppliedKernels,
												uint const		doNotApplyWith
												);
}

SteerForSeparationCUDA::SteerForSeparationCUDA(	AgentGroup * pAgentGroup, KNNData * pKNNData, AgentGroup * pOtherGroup, float const fWeight, uint const doNotApplyWith )
:	AbstractCUDAKernel( pAgentGroup, fWeight, doNotApplyWith ),
	m_pKNNData( pKNNData ),
	m_pOtherGroup( pOtherGroup )
{
	// Nothing to do.
}

void SteerForSeparationCUDA::init( void )
{
	// Nothing to do.
}

void SteerForSeparationCUDA::run( void )
{
	dim3 grid = gridDim();
	dim3 block = blockDim();

	size_t const	numAgents			= getNumAgents();
	size_t const	k					= m_pKNNData->k();

	// Gather required device pointers.
	float3 const*	pdPosition			= m_pAgentGroupData->pdPosition();
	float3 *		pdSteering			= m_pAgentGroupData->pdSteering();
	uint const*		pdKNNIndices		= m_pKNNData->pdKNNIndices();

	uint *			pdAppliedKernels	= m_pAgentGroupData->pdAppliedKernels();

	// Compute the size of shared memory needed for each block.
	size_t shMemSize = k * THREADSPERBLOCK * sizeof(uint);

	// Launch the kernel.
	SteerForSeparationKernel<<< grid, block, shMemSize >>>( pdKNNIndices, k, pdPosition, pdSteering, numAgents, m_fWeight, pdAppliedKernels, m_doNotApplyWith );
	cutilCheckMsg( "SteerForSeparationKernel failed." );
	CUDA_SAFE_CALL( cudaThreadSynchronize() );
}

void SteerForSeparationCUDA::close( void )
{
	// The AgentGroup data has most likely changed.
	m_pAgentGroup->SetSyncHost();
}
