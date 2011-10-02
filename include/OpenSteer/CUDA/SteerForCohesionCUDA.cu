#include "SteerForCohesionCUDA.cuh"

extern "C"
{
	__global__ void SteerForCohesionCUDAKernel(	float3 const*	pdAPosition,
												float3 const*	pdADirection,
												float3 *		pdASteering,
												size_t const	numA,

												uint const*		pdKNNIndices,
												size_t const	k,

												float3 const*	pdBPosition,
												uint const		numB,

												float const		maxDistance,
												float const		cosMaxAngle,

												float const		fWeight,
												uint *			pdAppliedKernels,
												uint const		doNotApplyWith
												);
}

using namespace OpenSteer;

SteerForCohesionCUDA::SteerForCohesionCUDA(	AgentGroup * pAgentGroup, KNNData * pKNNData, AgentGroup * pOtherGroup, float const maxDistance, float const cosMaxAngle, float const fWeight, uint const doNotApplyWith )
:	AbstractCUDAKernel( pAgentGroup, fWeight, doNotApplyWith ),
	m_pOtherGroup( pOtherGroup ),
	m_pKNNData( pKNNData ),
	m_fCosMaxAngle( cosMaxAngle ),
	m_fMaxDistance( maxDistance )
{
	// Nothing to do.
}

void SteerForCohesionCUDA::init( void )
{
	// Nothing to do.
}

void SteerForCohesionCUDA::run( void )
{
	dim3 grid = gridDim();
	dim3 block = blockDim();

	// Gather the required device pointers.
	float3 const*	pdAPosition			= m_pAgentGroupData->pdPosition();
	float3 const*	pdADirection		= m_pAgentGroupData->pdDirection();
	float3 *	pdASteering				= m_pAgentGroupData->pdSteering();
	
	uint const&		numA				= getNumAgents();

	uint const*		pdKNNIndices		= m_pKNNData->pdKNNIndices();
	uint const&		k					= m_pKNNData->k();

	float3 const*	pdBPosition			= m_pOtherGroup->pdPosition();
	uint const&		numB				= m_pOtherGroup->Size();

	uint *			pdAppliedKernels	= m_pAgentGroupData->pdAppliedKernels();

	size_t const	shMemSize			= THREADSPERBLOCK * k * sizeof(uint);

	SteerForCohesionCUDAKernel<<< grid, block, shMemSize >>>(	// Agent data.
																pdAPosition,
																pdADirection,
																pdASteering,
																numA,
																// KNN data.
																pdKNNIndices,
																k,
																// Other group data.
																pdBPosition,
																numB,
																// Flocking data.
																m_fMaxDistance,
																m_fCosMaxAngle,
														
																m_fWeight,
																pdAppliedKernels,
																m_doNotApplyWith
																);
	cutilCheckMsg( "SteerForCohesionCUDAKernel failed" );
	CUDA_SAFE_CALL( cudaThreadSynchronize() );
}

void SteerForCohesionCUDA::close( void )
{
	m_pAgentGroup->SetSyncHost();
}
