#include "AvoidObstaclesCUDA.cuh"

using namespace OpenSteer;

// Kernel function prototype.
extern "C"
{
	__host__ void SteerToAvoidObstaclesKernelBindTextures(	float4 const*	pdObstaclePosition,
															float const*	pdObstacleRadius,
															uint const		numObstacles
															);
	__host__ void SteerToAvoidObstaclesKernelUnbindTextures( void );

	__global__ void SteerToAvoidObstaclesKernel(			uint const*		pdKNNIndices,			// In:	Indices of the K Nearest Obstacles.
															float const*	pdKNNDistances,			// In:	Distances to the K Nearest Obstacles.
															size_t const	k,
														
															float4 const*	pdPosition,				// In:	Agent positions.
															float4 const*	pdDirection,			// In:	Agent directions.
															float3 const*	pdSide,
															float3 const*	pdUp,
															float const*	pdRadius,				// In:	Agent radii.
															float const*	pdSpeed,				// In:	Agent speeds.

															float const		minTimeToCollision,
				
															float4 *		pdSteering,				// Out:	Agent steering vectors.
															
															uint const		numAgents,				// In:	Number of agents.
															uint const		numObstacles,			// In:	Number of obstacles.
															float const		fWeight,				// In:	Weight for this kernel

															uint *			pdAppliedKernels,
															uint const		doNotApplyWith
															);
}

AvoidObstaclesCUDA::AvoidObstaclesCUDA( AgentGroup * pAgentGroup, ObstacleGroup * pObstacleGroup, KNNData * pKNNData, float const fMinTimeToCollision, float const fWeight, uint const doNotApplyWith )
:	AbstractCUDAKernel( pAgentGroup, fWeight, doNotApplyWith ),
	m_pObstacleGroup( pObstacleGroup ),
	m_fMinTimeToCollision( fMinTimeToCollision ),
	m_pKNNData( pKNNData )
{
	// Nothing to do.
}

void AvoidObstaclesCUDA::init(void)
{

}

void AvoidObstaclesCUDA::run(void)
{
	dim3 grid = gridDim();
	dim3 block = blockDim();

	uint const*		pdKNNIndices		= m_pKNNData->pdKNNIndices();
	float const*	pdKNNDistances		= m_pKNNData->pdKNNDistances();
	uint const&		k					= m_pKNNData->k();

	float4 const*	pdPosition			= m_pAgentGroupData->pdPosition();
	float4 const*	pdDirection			= m_pAgentGroupData->pdDirection();
	float3 const*	pdSide				= m_pAgentGroupData->pdSide();
	float3 const*	pdUp				= m_pAgentGroupData->pdUp();
	float const*	pdRadius			= m_pAgentGroupData->pdRadius();
	float const*	pdSpeed				= m_pAgentGroupData->pdSpeed();

	float4 const*	pdObstaclePosition	= m_pObstacleGroup->pdPosition();
	float const*	pdObstacleRadius	= m_pObstacleGroup->pdRadius();

	float4 *		pdSteering			= m_pAgentGroupData->pdSteering();
	
	uint const&		numAgents			= m_pAgentGroup->Size();
	uint const&		numObstacles		= m_pObstacleGroup->Size();

	uint *			pdAppliedKernels	= m_pAgentGroupData->pdAppliedKernels();

	size_t shMemSize = k * THREADSPERBLOCK * (sizeof(uint) + sizeof(float));

	// Bind the textures.
	SteerToAvoidObstaclesKernelBindTextures( pdObstaclePosition, pdObstacleRadius, numObstacles );

	SteerToAvoidObstaclesKernel<<< grid, block, shMemSize >>>(	pdKNNIndices, pdKNNDistances, k,
																pdPosition, pdDirection, pdSide, pdUp, pdRadius, pdSpeed,
																m_fMinTimeToCollision, 
																pdSteering, 
																numAgents, numObstacles,
																m_fWeight,
																pdAppliedKernels, m_doNotApplyWith
																);
	cutilCheckMsg( "AvoidObstaclesCUDAKernel failed." );
	CUDA_SAFE_CALL( cudaThreadSynchronize() );

	// Unbind the textures.
	SteerToAvoidObstaclesKernelUnbindTextures();
}

void AvoidObstaclesCUDA::close(void)
{
	// Device data has changed. Instruct the AgentGroup it needs to synchronize the host.
	m_pAgentGroup->SetSyncHost();
}
