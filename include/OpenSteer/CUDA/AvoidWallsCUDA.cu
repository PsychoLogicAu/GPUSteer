#include "AvoidWallsCUDA.cuh"

using namespace OpenSteer;

extern "C"
{
	
	__host__ void SteerToAvoidWallsKernelBindTextures(	float4 const*	pdLineStart,
														float4 const*	pdLineEnd,
														float4 const*	pdLineNormal,
														uint const		numLines
														);
	__host__ void SteerToAvoidWallsKernelUnbindTextures( void );

	__global__ void SteerToAvoidWallsCUDAKernel(		// Agent data.
														float4 const*	pdPosition,
														float4 const*	pdDirection,
														float3 const*	pdSide,
														float const*	pdSpeed,
														float const*	pdRadius,

														uint const*		pdKNLIndices,	// Indices of the K Nearest line segments...
														uint const		k,				// Number of lines in KNL.

														float const		minTimeToCollision,

														float4 *		pdSteering,

														uint const		numAgents,
														uint const		numLines,

														float const		fWeight,
														uint *			pdAppliedKernels,
														uint const		doNotApplyWith
														);
}

AvoidWallsCUDA::AvoidWallsCUDA( AgentGroup * pAgentGroup, KNNData * pKNNData, WallGroup * pWallGroup, float const fMinTimeToCollision, float const fWeight, uint const doNotApplyWith )
:	AbstractCUDAKernel( pAgentGroup, fWeight, doNotApplyWith ),
	m_pWallGroup( pWallGroup ),
	m_fMinTimeToCollision( fMinTimeToCollision ),
	m_pKNNData( pKNNData )
{
	// Nothing to do.
}

void AvoidWallsCUDA::init( void )
{
	// Nothing to do.
}

void AvoidWallsCUDA::run( void )
{
	dim3 grid = gridDim();
	dim3 block = blockDim();

	// Gather the required device pointers.
	float4 const*		pdPosition			= m_pAgentGroupData->pdPosition();
	float4 const*		pdDirection			= m_pAgentGroupData->pdDirection();
	float3 const*		pdSide				= m_pAgentGroupData->pdSide();
	float const*		pdSpeed				= m_pAgentGroupData->pdSpeed();
	float const*		pdRadius			= m_pAgentGroupData->pdRadius();

	float4 *			pdSteering			= m_pAgentGroupData->pdSteering();

	float4 const*		pdLineStart			= m_pWallGroup->GetWallGroupData().pdLineStart();
	float4 const*		pdLineEnd			= m_pWallGroup->GetWallGroupData().pdLineEnd();
	float4 const*		pdLineNormal		= m_pWallGroup->GetWallGroupData().pdLineNormal();
	uint const&			numLines			= m_pWallGroup->Size();

	uint const*			pdKNLIndices		= m_pKNNData->pdKNNIndices();

	uint const&			k					= m_pKNNData->k();
	uint const&			numAgents			= getNumAgents();

	uint *				pdAppliedKernels	= m_pAgentGroupData->pdAppliedKernels();

	size_t shMemSize = k * THREADSPERBLOCK * sizeof( uint );

	// Bind the textures.
	SteerToAvoidWallsKernelBindTextures( pdLineStart, pdLineEnd, pdLineNormal, numLines );

	SteerToAvoidWallsCUDAKernel<<<grid, block, shMemSize>>>(	pdPosition,
																pdDirection,
																pdSide,
																pdSpeed,
																pdRadius,

																pdKNLIndices,
																k,
																
																m_fMinTimeToCollision,
																pdSteering,
																numAgents,
																numLines,
																m_fWeight,
																pdAppliedKernels,
																m_doNotApplyWith
																);
	cutilCheckMsg( "AvoidWallsCUDAKernel failed" );
	CUDA_SAFE_CALL( cudaThreadSynchronize() );

	// Unbind the textures.
	SteerToAvoidWallsKernelUnbindTextures();
}

void AvoidWallsCUDA::close( void )
{
	// Agent group data may have changed.
	m_pAgentGroup->SetSyncHost();
}
