#include "SteerForAlignmentCUDA.cuh"

extern "C"
{
	__global__ void SteerForAlignmentCUDAKernel(	float3 const*	pdAPosition,
													float3 const*	pdADirection,
													float3 *		pdASteering,
													size_t const	numA,

													uint const*		pdKNNIndices,
													size_t const	k,

													float3 const*	pdBPosition,
													float3 const*	pdBDirection,
													uint const		numB,

													float const		maxDistance,
													float const		cosMaxAngle,

													float const		fWeight,
													uint *			pdAppliedKernels,
													uint const		doNotApplyWith
													);
}

using namespace OpenSteer;

SteerForAlignmentCUDA::SteerForAlignmentCUDA(	AgentGroup * pAgentGroup, KNNData * pKNNData, AgentGroup * pOtherGroup, float const maxDistance, float const cosMaxAngle, float const fWeight, uint const doNotApplyWith )
:	AbstractCUDAKernel( pAgentGroup, fWeight, doNotApplyWith ),
	m_pKNNData( pKNNData ),
	m_pOtherGroup( pOtherGroup ),
	m_fMaxDistance( maxDistance ),
	m_fCosMaxAngle( cosMaxAngle )
{
	// Nothing to do.
}

void SteerForAlignmentCUDA::init( void )
{
	// Nothing to do.
}

void SteerForAlignmentCUDA::run( void )
{
	dim3 grid = gridDim();
	dim3 block = blockDim();

	float3 const*		pdAPosition			= m_pAgentGroupData->pdPosition();
	float3 const*		pdADirection		= m_pAgentGroupData->pdDirection();
	float3 *			pdASteering			= m_pAgentGroupData->pdSteering();
	uint const&			numA				= getNumAgents();

	uint const*			pdKNNIndices		= m_pKNNData->pdKNNIndices();
	uint const&			k					= m_pKNNData->k();

	float3 const*		pdBPosition			= m_pOtherGroup->pdPosition();
	float3 const*		pdBDirection		= m_pOtherGroup->pdDirection();
	uint const&			numB				= m_pOtherGroup->Size();

	uint *				pdAppliedKernels	= m_pAgentGroupData->pdAppliedKernels();

	size_t const		shMemSize			= k * THREADSPERBLOCK * sizeof(uint);

	SteerForAlignmentCUDAKernel<<< grid, block, shMemSize >>>(	pdAPosition,
																pdADirection,
																pdASteering,
																numA,
																
																pdKNNIndices,
																k,

																pdBPosition,
																pdBDirection,
																numB,

																m_fMaxDistance,
																m_fCosMaxAngle,
																
																m_fWeight,

																pdAppliedKernels,
																m_doNotApplyWith
																);
	cutilCheckMsg( "SteerForAlignmentCUDAKernel failed" );
	CUDA_SAFE_CALL( cudaThreadSynchronize() );
}

void SteerForAlignmentCUDA::close( void )
{
	m_pAgentGroup->SetSyncHost();
}
