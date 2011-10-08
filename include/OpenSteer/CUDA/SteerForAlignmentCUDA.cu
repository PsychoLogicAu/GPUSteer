#include "SteerForAlignmentCUDA.cuh"

extern "C"
{
	__host__ void SteerForAlignmentKernelBindTextures(	float4 const*	pdBPosition,
														float4 const*	pdBDirection,
														uint const		numB
														);
	__host__ void SteerForAlignmentKernelUnindTextures( void );

	__global__ void SteerForAlignmentCUDAKernel(		float4 const*	pdPosition,
														float4 const*	pdDirection,
														float4 *		pdSteering,
														size_t const	numA,

														uint const*		pdKNNIndices,
														size_t const	k,

														uint const		numB,

														float const		minDistance,
														float const		maxDistance,
														float const		cosMaxAngle,

														float const		fWeight,
														uint *			pdAppliedKernels,
														uint const		doNotApplyWith
														);
}

using namespace OpenSteer;

SteerForAlignmentCUDA::SteerForAlignmentCUDA(	AgentGroup * pAgentGroup, KNNData * pKNNData, AgentGroup * pOtherGroup, float const minDistance, float const maxDistance, float const cosMaxAngle, float const fWeight, uint const doNotApplyWith )
:	AbstractCUDAKernel( pAgentGroup, fWeight, doNotApplyWith ),
	m_pKNNData( pKNNData ),
	m_pOtherGroup( pOtherGroup ),
	m_fMinDistance( minDistance ),
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

	float4 const*		pdAPosition			= m_pAgentGroupData->pdPosition();
	float4 const*		pdADirection		= m_pAgentGroupData->pdDirection();
	float4 *			pdASteering			= m_pAgentGroupData->pdSteering();
	uint const&			numA				= getNumAgents();

	uint const*			pdKNNIndices		= m_pKNNData->pdKNNIndices();
	uint const&			k					= m_pKNNData->k();

	float4 const*		pdBPosition			= m_pOtherGroup->pdPosition();
	float4 const*		pdBDirection		= m_pOtherGroup->pdDirection();
	uint const&			numB				= m_pOtherGroup->Size();

	uint *				pdAppliedKernels	= m_pAgentGroupData->pdAppliedKernels();

	size_t const		shMemSize			= k * THREADSPERBLOCK * sizeof(uint);

	// Bind the textures.
	SteerForAlignmentKernelBindTextures( pdBPosition, pdBDirection, numB );

	SteerForAlignmentCUDAKernel<<< grid, block, shMemSize >>>(	pdAPosition,
																pdADirection,
																pdASteering,
																numA,
																
																pdKNNIndices,
																k,

																numB,

																m_fMinDistance,
																m_fMaxDistance,
																m_fCosMaxAngle,
																
																m_fWeight,

																pdAppliedKernels,
																m_doNotApplyWith
																);
	cutilCheckMsg( "SteerForAlignmentCUDAKernel failed" );
	CUDA_SAFE_CALL( cudaThreadSynchronize() );

	// Unbind the textures.
	SteerForAlignmentKernelUnindTextures();
}

void SteerForAlignmentCUDA::close( void )
{
	// Agent group data may have changed.
	m_pAgentGroup->SetSyncHost();
}
