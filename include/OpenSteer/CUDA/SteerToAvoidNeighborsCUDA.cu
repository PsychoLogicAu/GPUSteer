#include "SteerToAvoidNeighborsCUDA.cuh"

using namespace OpenSteer;

extern "C"
{
	__global__ void SteerToAvoidNeighborsCUDAKernel(	uint const*		pdKNNIndices,			// In:		Indices of the KNN for each agent.
														float const*	pdKNNDistances,			// In:		Distances to the KNN for each agent.
														size_t const	k,						// In:		Number of KNN for each agent.

														float3 const*	pdPosition,				// In:		Positions of each agent.
														float3 const*	pdDirection,			// In:		Directions of facing for each agent.
														float const*	pdRadius,				// In:		Radius of each agent.
														float3 const*	pdSide,					// In:		Side direction for each agent.

														float *			pdSpeed,				// In/Out:	Speed of each agent.
														float3 *		pdSteering,				// Out:		Steering vectors for each agent.

														float const		minTimeToCollision,		// In:		Look-ahead time for collision avoidance.
														float const		minSeparationDistance,	// In:		Distance to consider 'close' neighbors.

														size_t const	numAgents,
														float const		fWeight
														);

	__global__ void SteerToAvoidCloseNeighborsCUDAKernel(	uint const*		pdKNNIndices,
															float const*	pdKNNDistances,
															size_t const	k,

															float3 const*	pdPosition,
															float3 const*	pdDirection,
															float const*	pdRadius,

															float3 *		pdSteering,
															float *			pdSpeed,

															float const		minSeparationDistance,

															size_t const	numAgents,
															float const		fWeight
															);
}

SteerToAvoidNeighborsCUDA::SteerToAvoidNeighborsCUDA( AgentGroup * pAgentGroup, KNNData * pKNNData, AgentGroup * pOtherGroup, float const fMinTimeToCollision, float const fMinSeparationDistance, float const fWeight )
:	AbstractCUDAKernel( pAgentGroup, fWeight ),
	m_pKNNData( pKNNData ),
	m_pOtherGroup( pOtherGroup ),
	m_fMinTimeToCollision( fMinTimeToCollision ),
	m_fMinSeparationDistance( fMinSeparationDistance )
{
}

void SteerToAvoidNeighborsCUDA::init( void )
{
	// Nothing to do.
}

void SteerToAvoidNeighborsCUDA::run( void )
{
	dim3 grid = gridDim();
	dim3 block = blockDim();

	size_t const&	k				= m_pKNNData->k();
	size_t const&	numAgents		= getNumAgents();

	// Gather the required device pointers.
	uint const*		pdKNNIndices	= m_pKNNData->pdKNNIndices();
	float const*	pdKNNDistances	= m_pKNNData->pdKNNDistances();

	float3 const*	pdPosition		= m_pAgentGroupData->pdPosition();
	float3 const*	pdDirection		= m_pAgentGroupData->pdForward();
	float3 const*	pdSide			= m_pAgentGroupData->pdSide();
	float *			pdSpeed			= m_pAgentGroupData->pdSpeed();

	float const*	pdRadius		= m_pAgentGroupConst->pdRadius();

	float3 *		pdSteering		= m_pAgentGroupData->pdSteering();

	// Only consider to be 'close neighbor' when this close.
	m_fMinSeparationDistance = 1.f;

	//
	// Avoid the 'close' neighbors.
	//

	// Compute the size of shared memory needed for each block.
	//size_t shMemSize = k * THREADSPERBLOCK * (sizeof(uint) + sizeof(float));
	//
	//SteerToAvoidCloseNeighborsCUDAKernel<<< grid, block, shMemSize >>>( pdKNNIndices, pdKNNDistances, k, pdPosition, pdDirection, pdRadius, pdSteering, pdSpeed, m_fMinSeparationDistance, numAgents );
	//cutilCheckMsg( "SteerToAvoidCloseNeighborsCUDAKernel failed." );
	//CUDA_SAFE_CALL( cudaThreadSynchronize() );

	// Compute the size of shared memory needed for each block.
	size_t shMemSize = k * THREADSPERBLOCK * (sizeof(uint) + sizeof(float));
	
	SteerToAvoidNeighborsCUDAKernel<<< grid, block, shMemSize >>>( pdKNNIndices, pdKNNDistances, k, pdPosition, pdDirection, pdRadius, pdSide, pdSpeed, pdSteering, m_fMinTimeToCollision, m_fMinSeparationDistance, numAgents, m_fWeight );
	cutilCheckMsg( "SteerToAvoidNeighborsCUDAKernel failed." );
	CUDA_SAFE_CALL( cudaThreadSynchronize() );
}

void SteerToAvoidNeighborsCUDA::close(void )
{
	// Device data has changed. Instruct the AgentGroup it needs to synchronize the host.
	m_pAgentGroup->SetSyncHost();
}
