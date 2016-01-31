#include "SteerToAvoidNeighborsCUDA.cuh"

using namespace OpenSteer;

extern "C"
{
	__host__ void SteerToAvoidNeighborsKernelUnbindTextures( void );
	__host__ void SteerToAvoidNeighborsKernelBindTextures(	float4 const*	pdBPosition,
															float4 const*	pdBDirection,
															float const*	pdBSpeed,
															float const*	pdBRadius,
															uint const		numB
															);

	__global__ void SteerToAvoidNeighborsCUDAKernel(		uint const*		pdKNNIndices,			// In:		Indices of the KNN for each agent.
															float const*	pdKNNDistances,			// In:		Distances to the KNN for each agent.
															size_t const	k,						// In:		Number of KNN for each agent.

															// Group A data.
															float4 const*	pdPosition,				// In:		Positions of each agent.
															float4 const*	pdDirection,			// In:		Directions of facing for each agent.
															float const*	pdRadius,				// In:		Radius of each agent.
															float3 const*	pdSide,					// In:		Side direction for each agent.

															float *			pdSpeed,				// In/Out:	Speed of each agent.
															float4 *		pdSteering,				// Out:		Steering vectors for each agent.
															uint const		numA,

															// Group B data.
															uint const		numB,

															float const		minTimeToCollision,		// In:		Look-ahead time for collision avoidance.
															float const		minSeparationDistance,	// In:		Distance to consider 'close' neighbors.

															float const		fWeight,

															uint *			pdAppliedKernels,
															uint const		doNotApplyWith
															);
}

SteerToAvoidNeighborsCUDA::SteerToAvoidNeighborsCUDA( AgentGroup * pAgentGroup, KNNData * pKNNData, AgentGroup * pOtherGroup, float const fMinTimeToCollision, float const fMinSeparationDistance, float const fWeight, uint const doNotApplyWith )
:	AbstractCUDAKernel( pAgentGroup, fWeight, doNotApplyWith ),
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

	size_t const&	k					= m_pKNNData->k();
	

	// Gather the required device pointers.
	uint const*		pdKNNIndices		= m_pKNNData->pdKNNIndices();
	float const*	pdKNNDistances		= m_pKNNData->pdKNNDistances();

	float4 const*	pdAPosition			= m_pAgentGroupData->pdPosition();
	float4 const*	pdADirection		= m_pAgentGroupData->pdDirection();
	float3 const*	pdASide				= m_pAgentGroupData->pdSide();
	float *			pdASpeed			= m_pAgentGroupData->pdSpeed();
	float const*	pdARadius			= m_pAgentGroupData->pdRadius();
	float4 *		pdASteering			= m_pAgentGroupData->pdSteering();
	uint const&		numA				= getNumAgents();


	float4 const*	pdBPosition			= m_pOtherGroup->pdPosition();
	float4 const*	pdBDirection		= m_pOtherGroup->pdDirection();
	float const*	pdBSpeed			= m_pOtherGroup->pdSpeed();
	float const*	pdBRadius			= m_pOtherGroup->pdRadius();

	uint const&		numB				= m_pOtherGroup->Size();

	uint *			pdAppliedKernels	= m_pAgentGroupData->pdAppliedKernels();

	size_t shMemSize = k * THREADSPERBLOCK * (sizeof(uint) + sizeof(float));

	// Bind the textures.
	SteerToAvoidNeighborsKernelBindTextures( pdBPosition, pdBDirection, pdBSpeed, pdBRadius, numB );
	
	SteerToAvoidNeighborsCUDAKernel<<< grid, block, shMemSize >>>(	pdKNNIndices, pdKNNDistances, k,
																	pdAPosition, pdADirection, pdARadius, pdASide,
																	pdASpeed, pdASteering,
																	numA,
																	numB,
																	m_fMinTimeToCollision,
																	m_fMinSeparationDistance,
																	m_fWeight,
																	pdAppliedKernels,
																	m_doNotApplyWith
																	);
	cutilCheckMsg( "SteerToAvoidNeighborsCUDAKernel failed." );
	CUDA_SAFE_CALL( cudaThreadSynchronize() );

	// Unbind the textures.
	SteerToAvoidNeighborsKernelUnbindTextures();
}

void SteerToAvoidNeighborsCUDA::close(void )
{
	// Device data has changed. Instruct the AgentGroup it needs to synchronize the host.
	m_pAgentGroup->SetSyncHost();
}
