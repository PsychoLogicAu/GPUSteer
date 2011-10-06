#include "SteerForSeparationCUDA.cuh"

using namespace OpenSteer;

extern "C"
{
	__global__ void SteerForSeparationKernel(	float3 const*	pdAPosition,
												float3 const*	pdADirection,
												float3 *		pdASteering,
												size_t const	numA,

												uint const*		pdKNNIndices,
												size_t const	k,

												float3 const*	pdBPosition,
												uint const		numB,

												float const		minDistance,
												float const		maxDistance,
												float const		cosMaxAngle,

												float const		fWeight,
												uint *			pdAppliedKernels,
												uint const		doNotApplyWith
												);
}

SteerForSeparationCUDA::SteerForSeparationCUDA(	AgentGroup * pAgentGroup, KNNData * pKNNData, AgentGroup * pOtherGroup, float const minDistance, float const maxDistance, float const cosMaxAngle, float const fWeight, uint const doNotApplyWith )
:	AbstractCUDAKernel( pAgentGroup, fWeight, doNotApplyWith ),
	m_pKNNData( pKNNData ),
	m_pOtherGroup( pOtherGroup ),
	m_fMinDistance( minDistance ),
	m_fMaxDistance( maxDistance ),
	m_fCosMaxAngle( cosMaxAngle )
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

	// Gather required device pointers.
	float3 const*	pdAPosition			= m_pAgentGroupData->pdPosition();
	float3 const*	pdADirection		= m_pAgentGroupData->pdDirection();
	float3 *		pdASteering			= m_pAgentGroupData->pdSteering();
	size_t const	numA				= getNumAgents();

	uint const*		pdKNNIndices		= m_pKNNData->pdKNNIndices();
	size_t const&	k					= m_pKNNData->k();

	uint *			pdAppliedKernels	= m_pAgentGroupData->pdAppliedKernels();

	float3 const*	pdBPosition			= m_pOtherGroup->pdPosition();
	uint const&		numB				= m_pOtherGroup->Size();

	// Compute the size of shared memory needed for each block.
	size_t shMemSize = k * THREADSPERBLOCK * sizeof(uint);

	// Launch the kernel.
	SteerForSeparationKernel<<< grid, block, shMemSize >>>(	pdAPosition, 
															pdADirection, 
															pdASteering, 
															numA, 
															pdKNNIndices, 
															k, 
															pdBPosition, 
															numB, 
															m_fMinDistance,
															m_fMaxDistance, 
															m_fCosMaxAngle, 
															m_fWeight, 
															pdAppliedKernels, 
															m_doNotApplyWith 
															);
	cutilCheckMsg( "SteerForSeparationKernel failed." );
	CUDA_SAFE_CALL( cudaThreadSynchronize() );
}

void SteerForSeparationCUDA::close( void )
{
	// The AgentGroup data has most likely changed.
	m_pAgentGroup->SetSyncHost();
}
