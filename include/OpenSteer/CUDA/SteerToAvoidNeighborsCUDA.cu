#include "SteerToAvoidNeighborsCUDA.cuh"

using namespace OpenSteer;

extern "C"
{
	__host__ void SteerToAvoidNeighborsKernelBindTextures(	float4 const*	pdBPosition,
															float4 const*	pdBDirection,
															float const*	pdBSpeed,
															float const*	pdBRadius,
															uint const		numB
															);
	__host__ void SteerToAvoidNeighborsKernelUnbindTextures( void );

	__global__ void SteerToAvoidNeighborsCUDAKernel(		// KNN data.
															uint const*		pdKNNIndices,
															size_t const	k,
															
															// Group A data.
															float4 const*	pdPosition,
															float4 const*	pdDirection,
															float3 const*	pdSide,
															float const*	pdRadius,

															float const*	pdSpeed,
															float4 *		pdSteering,
															size_t const	numA,

															// Group B data.
															uint const		numB,

															float const		minTimeToCollision,

															float const		fWeight,

															uint *			pdAppliedKernels,
															uint const		doNotApplyWith
															);

	__global__ void SteerToAvoidCloseNeighborsCUDAKernel(	// KNN data.
															uint const*		pdKNNIndices,
															float const*	pdKNNDistances,
															size_t const	k,

															// Group A data.
															float4 const*	pdPosition,
															float4 const*	pdDirection,
															float const*	pdRadius,

															float4 *		pdSteering,
															size_t const	numA,

															// Group B data.
															uint const		numB,

															float const		minSeparationDistance,

															float const		fWeight,

															uint *			pdAppliedKernels,
															uint const		doNotApplyWith
															);
}




SteerToAvoidNeighborsCUDA::SteerToAvoidNeighborsCUDA( AgentGroup * pAgentGroup, KNNData * pKNNData, AgentGroup * pOtherGroup, float const fMinTimeToCollision, float const fMinSeparationDistance, bool const bAvoidCloseNeighbors, float const fWeight, uint const doNotApplyWith )
:	AbstractCUDAKernel( pAgentGroup, fWeight, doNotApplyWith ),
	m_bAvoidCloseNeighbors( bAvoidCloseNeighbors ),
	m_fMinTimeToCollision( fMinTimeToCollision ),
	m_fMinSeparationDistance( fMinSeparationDistance ),
	m_pKNNData( pKNNData ),
	m_pOtherGroup( pOtherGroup )
{
	// Nothing to do.
}

void SteerToAvoidNeighborsCUDA::init( void )
{
	// Nothing to do.
}

void SteerToAvoidNeighborsCUDA::run( void )
{
	dim3 grid = gridDim();
	dim3 block = blockDim();

	// KNN data.
	uint const*		pdKNNIndices		= m_pKNNData->pdKNNIndices();
	float const*	pdKNNDistances		= m_pKNNData->pdKNNDistances();
	uint const&		k					= m_pKNNData->k();

	// Group A data.
	float4 const*	pdPosition			= m_pAgentGroupData->pdPosition();
	float4 const*	pdDirection			= m_pAgentGroupData->pdDirection();
	float3 const*	pdSide				= m_pAgentGroupData->pdSide();
	float const*	pdRadius			= m_pAgentGroupData->pdRadius();
	float const*	pdSpeed				= m_pAgentGroupData->pdSpeed();
	float4 *		pdSteering			= m_pAgentGroupData->pdSteering();
	uint const&		numA				= m_pAgentGroup->Size();
	uint *			pdAppliedKernels	= m_pAgentGroupData->pdAppliedKernels();

	// Group B data.
	float4 const*	pdBPosition			= m_pOtherGroup->pdPosition();
	float4 const*	pdBDirection		= m_pOtherGroup->pdDirection();
	float const*	pdBSpeed			= m_pOtherGroup->pdSpeed();
	float const*	pdBRadius			= m_pOtherGroup->pdRadius();
	uint const&		numB				= m_pOtherGroup->Size();

	size_t 			shMemSize;

	// Bind the textures.
	SteerToAvoidNeighborsKernelBindTextures( pdBPosition, pdBDirection, pdBSpeed, pdBRadius, numB );

	if( m_bAvoidCloseNeighbors )
	{
		shMemSize = THREADSPERBLOCK * k * (sizeof(uint) + sizeof(float));

		SteerToAvoidCloseNeighborsCUDAKernel<<< grid, block, shMemSize >>>(	pdKNNIndices,
																			pdKNNDistances,
																			k,

																			pdPosition,
																			pdDirection,
																			pdRadius,
																			pdSteering,
																			numA,

																			numB,
																			m_fMinSeparationDistance,
																			m_fWeight,
																			pdAppliedKernels,
																			m_doNotApplyWith
																			);
		cutilCheckMsg( "SteerToAvoidCloseNeighborsCUDAKernel failed." );
		CUDA_SAFE_CALL( cudaThreadSynchronize() );
	}

	shMemSize = THREADSPERBLOCK * k * sizeof(uint);

	SteerToAvoidNeighborsCUDAKernel<<< grid, block, shMemSize >>>(	pdKNNIndices,
																	k,

																	pdPosition,
																	pdDirection,
																	pdSide,
																	pdRadius,
																	pdSpeed,
																	pdSteering,
																	numA,

																	numB,

																	m_fMinTimeToCollision,

																	m_fWeight,
																	pdAppliedKernels,
																	m_doNotApplyWith | KERNEL_AVOID_CLOSE_NEIGHBORS_BIT
																	);
	cutilCheckMsg( "SteerToAvoidNeighborsCUDAKernel failed." );
	CUDA_SAFE_CALL( cudaThreadSynchronize() );

	// Unbind the textures.
	SteerToAvoidNeighborsKernelUnbindTextures();
}

void SteerToAvoidNeighborsCUDA::close(void )
{
	// Device data has changed. Instruct the VehicleGroup it needs to synchronize the host.
	m_pAgentGroup->SetSyncHost();
}
