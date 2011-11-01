#include "SteerToFollowPathCUDA.cuh"

extern "C"
{
	__global__ void FollowPathCUDAKernel(	// Agent data.
											float4 const*	pdPosition,
											float4 const*	pdDirection,
											float const*	pdSpeed,

											float4 *		pdSteering,

											// Path data.
											float3 const*	pdPathPoints,
											float const*	pdPathLengths,
											float3 const*	pdPathNormals,
											uint const		numPoints,
											float const		radius,
											bool const		cyclic,
											float const		totalPathLength,

											float const		predictionTime,

											uint const		numAgents,
											float const		fWeight,
											uint *			pdAppliedKernels,
											uint const		doNotApplyWith
											);
}

using namespace OpenSteer;

SteerToFollowPathCUDA::SteerToFollowPathCUDA( AgentGroup *pAgentGroup, PolylinePathwayCUDA * pPath, float const predictionTime, float const fWeight, uint const doNotApplyWith )
:	AbstractCUDAKernel( pAgentGroup, fWeight, doNotApplyWith ),
	m_pPath( pPath ),
	m_fPredictionTime( predictionTime )
{
	// Nothing to do.
}

void SteerToFollowPathCUDA::init( void )
{
	// Nothing to do.
}

void SteerToFollowPathCUDA::run( void )
{
	dim3 grid = gridDim();
	dim3 block = blockDim();

	float4 const*	pdPosition			= m_pAgentGroupData->pdPosition();
	float4 const*	pdDirection			= m_pAgentGroupData->pdDirection();
	float const*	pdSpeed				= m_pAgentGroupData->pdSpeed();

	float4 *		pdSteering			= m_pAgentGroupData->pdSteering();

	float3 const*	pdPathPoints		= m_pPath->pdPoints();
	float const*	pdPathLengths		= m_pPath->pdLengths();
	float3 const*	pdPathNormals		= m_pPath->pdNormals();
	uint const&		numPoints			= m_pPath->numPoints();
	float const&	radius				= m_pPath->radius();
	bool const&		cyclic				= m_pPath->cyclic();
	float const&	totalPathLength		= m_pPath->totalPathLength();

	uint const&		numAgents			= getNumAgents();

	uint *			pdAppliedKernels	= m_pAgentGroupData->pdAppliedKernels();

	FollowPathCUDAKernel<<< grid, block >>>(	// Agent data.
												pdPosition, pdDirection, pdSpeed,
												pdSteering,

												// Path data.
												pdPathPoints, pdPathLengths, pdPathNormals,
												numPoints, radius, cyclic, totalPathLength, 

												m_fPredictionTime,
												numAgents,
												m_fWeight,
												pdAppliedKernels,
												m_doNotApplyWith
												);
	cutilCheckMsg( "FollowPathCUDAKernel failed" );
	//CUDA_SAFE_CALL( cudaThreadSynchronize() );
}

void SteerToFollowPathCUDA::close(void )
{
	// Device data has changed. Instruct the AgentGroup it needs to synchronize the host.
	m_pAgentGroup->SetSyncHost();
}
