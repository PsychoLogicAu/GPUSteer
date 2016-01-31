#include "UpdateCUDA.h"

using namespace OpenSteer;

// Kernel function prototype.
extern "C"
{
	__global__ void UpdateCUDAKernel(	float3 * pdSide,
										float3 * pdUp,
										float4 * pdDirection,
										float4 * pdPosition,

										float4 * pdSteering,
										float * pdSpeed,

										float const* pdMaxForce,
										float const* pdMaxSpeed,
										float const* pdMass,

										float const elapsedTime,
										size_t const numAgents,
										uint * pdAppliedKernels
										);

	// New version.
	__host__ void UpdateCUDAKernelBindTextures(	float4 const*	pdLineStart,
												float4 const*	pdLineEnd,
												float4 const*	pdLineNormal,
												uint const		numLines
												);

	__host__ void UpdateCUDAKernelUnbindTextures( void );

	__global__ void UpdateCUDAKernelNew(		float3 *		pdSide,
												float3 *		pdUp,
												float4 *		pdDirection,
												float4 *		pdPosition,

												float4 *		pdSteering,
												float *			pdSpeed,

												float const*	pdMaxForce,
												float const*	pdMaxSpeed,
												float const*	pdMass,
												float const*	pdRadius,

												uint const*		pdKNLIndices,	// Indices of the K Nearest line segments...
												uint const		k,				// Number of lines in KNL.
												uint const		numLines,

												float const		elapsedTime,
												uint const		numAgents,
												uint *			pdAppliedKernels
												);
}

UpdateCUDA::UpdateCUDA( AgentGroup * pAgentGroup, const float fElapsedTime )
:	AbstractCUDAKernel( pAgentGroup, 1.f, 0 ),
	m_fElapsedTime( fElapsedTime )
{
}

void UpdateCUDA::init( void )
{
	// Nothing to do.
}

void UpdateCUDA::run(void)
{
	dim3 grid = gridDim();
	dim3 block = blockDim();

	// Gather pointers to the required data...
	float3 *		pdSide				= m_pAgentGroupData->pdSide();
	float3 *		pdUp				= m_pAgentGroupData->pdUp();
	float4 *		pdDirection			= m_pAgentGroupData->pdDirection();
	float4 *		pdPosition			= m_pAgentGroupData->pdPosition();
	float4 *		pdSteering			= m_pAgentGroupData->pdSteering();
	float *			pdSpeed				= m_pAgentGroupData->pdSpeed();

	float const*	pdMaxForce			= m_pAgentGroupData->pdMaxForce();
	float const*	pdMaxSpeed			= m_pAgentGroupData->pdMaxSpeed();
	float const*	pdMass				= m_pAgentGroupData->pdMass();
	float const*	pdRadius			= m_pAgentGroupData->pdRadius();

	uint *			pdAppliedKernels	= m_pAgentGroupData->pdAppliedKernels();

	uint const&		numAgents			= getNumAgents();

	UpdateCUDAKernel<<< grid, block >>>(	pdSide, pdUp, pdDirection, pdPosition, pdSteering, pdSpeed,
											pdMaxForce, pdMaxSpeed, pdMass,
											m_fElapsedTime, numAgents,
											pdAppliedKernels
											);
	cutilCheckMsg( "UpdateCUDAKernel failed." );
	//CUDA_SAFE_CALL( cudaThreadSynchronize() );

	// Unbind the textures.
	UpdateCUDAKernelUnbindTextures();
}

void UpdateCUDA::close(void)
{
	// Device data has changed. Instruct the AgentGroup it needs to synchronize the host.
	m_pAgentGroup->SetSyncHost();
}

#pragma region UpdateWithAntiPenetrationCUDA

UpdateWithAntiPenetrationCUDA::UpdateWithAntiPenetrationCUDA( AgentGroup * pAgentGroup, KNNData * pKNNData, WallGroup * pWallGroup, const float fElapsedTime )
:	AbstractCUDAKernel( pAgentGroup, 1.f, 0 ),
	m_fElapsedTime( fElapsedTime ),
	m_pKNNData( pKNNData ),
	m_pWallGroup( pWallGroup )
{
}

void UpdateWithAntiPenetrationCUDA::init( void )
{
	// Nothing to do.
}

void UpdateWithAntiPenetrationCUDA::run(void)
{
	dim3 grid = gridDim();
	dim3 block = blockDim();

	// Gather pointers to the required data...
	float3 *		pdSide				= m_pAgentGroupData->pdSide();
	float3 *		pdUp				= m_pAgentGroupData->pdUp();
	float4 *		pdDirection			= m_pAgentGroupData->pdDirection();
	float4 *		pdPosition			= m_pAgentGroupData->pdPosition();
	float4 *		pdSteering			= m_pAgentGroupData->pdSteering();
	float *			pdSpeed				= m_pAgentGroupData->pdSpeed();

	float const*	pdMaxForce			= m_pAgentGroupData->pdMaxForce();
	float const*	pdMaxSpeed			= m_pAgentGroupData->pdMaxSpeed();
	float const*	pdMass				= m_pAgentGroupData->pdMass();
	float const*	pdRadius			= m_pAgentGroupData->pdRadius();

	uint *			pdAppliedKernels	= m_pAgentGroupData->pdAppliedKernels();

	uint const&		numAgents			= getNumAgents();

	uint const*		pdKNLIndices		= m_pKNNData->pdKNNIndices();
	uint const&		k					= m_pKNNData->k();

	float4 const*	pdLineStart			= m_pWallGroup->GetWallGroupData().pdLineStart();
	float4 const*	pdLineEnd			= m_pWallGroup->GetWallGroupData().pdLineEnd();
	float4 const*	pdLineNormal		= m_pWallGroup->GetWallGroupData().pdLineNormal();
	uint const&		numLines			= m_pWallGroup->Size();

	size_t const	shMemSize			= k * THREADSPERBLOCK * sizeof(uint);

	// Bind the textures.
	UpdateCUDAKernelBindTextures( pdLineStart, pdLineEnd, pdLineNormal, numLines );

	UpdateCUDAKernelNew<<< grid, block, shMemSize >>>(	pdSide,
														pdUp,
														pdDirection,
														pdPosition,

														pdSteering,
														pdSpeed,
														pdMaxForce,
														pdMaxSpeed,
														pdMass,
														pdRadius,

														pdKNLIndices,
														k,
														numLines,

														m_fElapsedTime,
														numAgents,

														pdAppliedKernels
														);

	// Unbind the textures.
	UpdateCUDAKernelUnbindTextures();
}

void UpdateWithAntiPenetrationCUDA::close(void)
{
	// Device data has changed. Instruct the AgentGroup it needs to synchronize the host.
	m_pAgentGroup->SetSyncHost();
}
#pragma endregion
