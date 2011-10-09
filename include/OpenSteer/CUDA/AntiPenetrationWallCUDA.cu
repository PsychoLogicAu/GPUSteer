#include "AntiPenetrationWallCUDA.cuh"

extern "C"
{
	__host__ void AntiPenetrationWallKernelBindTextures(	float4 const*	pdLineStart,
															float4 const*	pdLineEnd,
															float4 const*	pdLineNormal,
															uint const		numLines
															);

	__host__ void AntiPenetrationWallKernelUnbindTextures( void );

	__global__ void AntiPenetrationWallKernel(				float4 const*	pdPosition,
															float4 *		pdDirection,
															float const*	pdSpeed,

															uint const*		pdKNLIndices,	// Indices of the K Nearest line segments...
															uint const		k,				// Number of lines in KNL.

															float const		elapsedTime,

															uint const		numAgents,
															uint const		numLines,
															uint *			pdAppliedKernels
															);
}

using namespace OpenSteer;

AntiPenetrationWALLCUDA::AntiPenetrationWALLCUDA( AgentGroup * pAgentGroup, KNNData * pKNNData, WallGroup * pWallGroup, float const elapsedTime, uint const doNotApplyWith )
:	AbstractCUDAKernel( pAgentGroup, 0.f, doNotApplyWith ),
	m_pKNNData( pKNNData ),
	m_pWallGroup( pWallGroup ),
	m_fElapsedTime( elapsedTime )
{

}

void AntiPenetrationWALLCUDA::init( void )
{
	// Nothing to do.
}

void AntiPenetrationWALLCUDA::run( void )
{
	dim3 grid = gridDim();
	dim3 block = blockDim();

	float4 const*		pdPosition			= m_pAgentGroupData->pdPosition();
	float4 *			pdDirection			= m_pAgentGroupData->pdDirection();
	float const*		pdSpeed				= m_pAgentGroupData->pdSpeed();
	uint const&			numAgents			= getNumAgents();

	uint const*			pdKNLIndices		= m_pKNNData->pdKNNIndices();
	uint const&			k					= m_pKNNData->k();

	uint *				pdAppliedKernels	= m_pAgentGroupData->pdAppliedKernels();

	float4 const*		pdLineStart			= m_pWallGroup->GetWallGroupData().pdLineStart();
	float4 const*		pdLineEnd			= m_pWallGroup->GetWallGroupData().pdLineEnd();
	float4 const*		pdLineNormal		= m_pWallGroup->GetWallGroupData().pdLineNormal();
	uint const&			numLines			= m_pWallGroup->Size();

	size_t const		shMemSize			= k * THREADSPERBLOCK * sizeof(uint);

	// Bind the textures.
	AntiPenetrationWallKernelBindTextures( pdLineStart, pdLineEnd, pdLineNormal, numLines );

	AntiPenetrationWallKernel<<< grid, block, shMemSize>>>(	pdPosition,
															pdDirection,
															pdSpeed,

															pdKNLIndices,
															k,

															m_fElapsedTime,
															numAgents,
															numLines,
															pdAppliedKernels
															);
	cutilCheckMsg( "AntiPenetrationWallKernel failed." );
	CUDA_SAFE_CALL( cudaThreadSynchronize() );

	// Unbind the textures.
	AntiPenetrationWallKernelUnbindTextures();
}

void AntiPenetrationWALLCUDA::close( void )
{
	// Agent group data may have changed.
	m_pAgentGroup->SetSyncHost();
}
