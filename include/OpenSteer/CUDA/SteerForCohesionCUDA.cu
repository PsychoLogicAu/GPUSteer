#include "SteerForCohesionCUDA.cuh"

extern "C"
{
	__host__ void SteerForCohesionKernelBindTextures(	float4 const*	pdBPosition,
														uint const		numB
														);
	__host__ void SteerForCohesionKernelUnindTextures( void );

	__global__ void SteerForCohesionCUDAKernel(			float4 const*	pdPosition,
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

SteerForCohesionCUDA::SteerForCohesionCUDA(	AgentGroup * pAgentGroup, KNNData * pKNNData, AgentGroup * pOtherGroup, float const minDistance, float const maxDistance, float const cosMaxAngle, float const fWeight, uint const doNotApplyWith )
:	AbstractCUDAKernel( pAgentGroup, fWeight, doNotApplyWith ),
	m_pOtherGroup( pOtherGroup ),
	m_pKNNData( pKNNData ),
	m_fCosMaxAngle( cosMaxAngle ),
	m_fMinDistance( minDistance ),
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
	float4 const*	pdAPosition			= m_pAgentGroupData->pdPosition();
	float4 const*	pdADirection		= m_pAgentGroupData->pdDirection();
	float4 *		pdASteering				= m_pAgentGroupData->pdSteering();
	
	uint const&		numA				= getNumAgents();

	uint const*		pdKNNIndices		= m_pKNNData->pdKNNIndices();
	uint const&		k					= m_pKNNData->k();

	float4 const*	pdBPosition			= m_pOtherGroup->pdPosition();
	uint const&		numB				= m_pOtherGroup->Size();

	uint *			pdAppliedKernels	= m_pAgentGroupData->pdAppliedKernels();

	size_t const	shMemSize			= THREADSPERBLOCK * k * sizeof(uint);

	// Bind the textures.
	SteerForCohesionKernelBindTextures( pdBPosition, numB );

	SteerForCohesionCUDAKernel<<< grid, block, shMemSize >>>(	// Agent data.
																pdAPosition,
																pdADirection,
																pdASteering,
																numA,
																// KNN data.
																pdKNNIndices,
																k,
																// Other group data.
																numB,

																// Flocking data.
																m_fMinDistance,
																m_fMaxDistance,
																m_fCosMaxAngle,
														
																m_fWeight,
																pdAppliedKernels,
																m_doNotApplyWith
																);
	cutilCheckMsg( "SteerForCohesionCUDAKernel failed" );
	CUDA_SAFE_CALL( cudaThreadSynchronize() );

	// Unbind the textures.
	SteerForCohesionKernelUnindTextures();
}

void SteerForCohesionCUDA::close( void )
{
	// Agent group data may have changed.
	m_pAgentGroup->SetSyncHost();
}
