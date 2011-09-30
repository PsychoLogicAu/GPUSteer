#include "AvoidWallsCUDA.cuh"

using namespace OpenSteer;

extern "C"
{
	__global__ void AvoidWallsCUDAKernel(	// Agent data.
											float3 const*	pdPosition,
											float3 const*	pdDirection,
											float3 const*	pdSide,
											float const*	pdSpeed,
											float const*	pdRadius,

											// Wall data.
											float3 const*	pdLineStart,
											float3 const*	pdLineEnd,
											float3 const*	pdLineNormal,

											uint const*		pdKNLIndices,	// Indices of the K Nearest line segments...
											uint const		k,				// Number of lines in KNL.

											float const		minTimeToCollision,

											float3 *		pdSteering,

											size_t const	numAgents,
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
	float3 const*		pdPosition			= m_pAgentGroupData->pdPosition();
	float3 const*		pdDirection			= m_pAgentGroupData->pdDirection();
	float3 const*		pdSide				= m_pAgentGroupData->pdSide();
	float const*		pdSpeed				= m_pAgentGroupData->pdSpeed();
	float const*		pdRadius			= m_pAgentGroupConst->pdRadius();

	float3 *			pdSteering			= m_pAgentGroupData->pdSteering();

	float3 const*		pdLineStart			= m_pWallGroup->GetWallGroupData().pdLineStart();
	float3 const*		pdLineEnd			= m_pWallGroup->GetWallGroupData().pdLineEnd();
	float3 const*		pdLineNormal		= m_pWallGroup->GetWallGroupData().pdLineNormal();

	uint const*			pdKNLIndices		= m_pKNNData->pdKNNIndices();

	uint const&			k					= m_pKNNData->k();
	uint const&			numAgents			= getNumAgents();

	uint *				pdAppliedKernels	= m_pAgentGroupData->pdAppliedKernels();

	size_t shMemSize = k * THREADSPERBLOCK * sizeof( uint );

	AvoidWallsCUDAKernel<<<grid, block, shMemSize>>>(	pdPosition, pdDirection, pdSide, pdSpeed, pdRadius,
														pdLineStart, pdLineEnd, pdLineNormal,
														pdKNLIndices,
														k, m_fMinTimeToCollision,
														pdSteering,
														numAgents, m_fWeight,
														pdAppliedKernels,
														m_doNotApplyWith );
	cutilCheckMsg( "AvoidWallsCUDAKernel failed" );
	CUDA_SAFE_CALL( cudaThreadSynchronize() );
}

void AvoidWallsCUDA::close( void )
{
	// Nothing to do.
}
