#include "SteerToAvoidNeighborsCUDA.cuh"

using namespace OpenSteer;

extern "C"
{
	__global__ void SteerToAvoidNeighborsCUDAKernel(		uint const*		pdKNNIndices,
															//float const*	pdKNNDistances,
															size_t const	k,
															
															float3 const*	pdPosition,
															float3 const*	pdDirection,
															float3 const*	pdSide,
															float const*	pdSpeed,

															float const*	pdRadius,

															float3 *		pdSteering,

															float const		minTimeToCollision,
															size_t const	numAgents
															);

	__global__ void SteerToAvoidCloseNeighborsCUDAKernel(	uint const*		pdKNNIndices,
															//float const*	pdKNNDistances,
															size_t const	k,

															float3 const*	pdPosition,
															float3 const*	pdDirection,
															float const*	pdRadius,

															float3 *		pdSteering,

															float const		minSeparationDistance,

															size_t const	numAgents
															);
}

SteerToAvoidNeighborsCUDA::SteerToAvoidNeighborsCUDA( VehicleGroup *pVehicleGroup, float const fMinTimeToCollision, float const fMinSeparationDistance )
:	AbstractCUDAKernel( pVehicleGroup ),
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

	size_t const&	k = m_pVehicleGroup->GetNearestNeighborData().k();
	size_t const&	numAgents = getNumAgents();

	// Gather the required device pointers.
	uint const*		pdKNNIndices = m_pVehicleGroup->GetNearestNeighborData().pdKNNIndices();
	float const*	pdKNNDistances = m_pVehicleGroup->GetNearestNeighborData().pdKNNDistances();

	float3 const*	pdPosition = m_pVehicleGroupData->pdPosition();
	float3 const*	pdDirection = m_pVehicleGroupData->pdForward();
	float3 const*	pdSide = m_pVehicleGroupData->pdSide();
	float const*	pdSpeed = m_pVehicleGroupData->pdSpeed();

	float const*	pdRadius = m_pVehicleGroupConst->pdRadius();

	float3 *		pdSteering = m_pVehicleGroupData->pdSteering();

	//
	// Avoid the 'close' neighbors.
	//

	// Compute the size of shared memory needed for each block.
	size_t shMemSize = k * THREADSPERBLOCK * sizeof(uint);
	
	SteerToAvoidCloseNeighborsCUDAKernel<<< grid, block, shMemSize >>>( pdKNNIndices, k, pdPosition, pdDirection, pdRadius, pdSteering, m_fMinSeparationDistance, numAgents );
	cutilCheckMsg( "SteerToAvoidCloseNeighborsCUDAKernel failed." );
	CUDA_SAFE_CALL( cudaThreadSynchronize() );

	// Compute the size of shared memory needed for each block.
	shMemSize = k * THREADSPERBLOCK * sizeof(uint);
	
	SteerToAvoidNeighborsCUDAKernel<<< grid, block, shMemSize >>>( pdKNNIndices, k, pdPosition, pdDirection, pdSide, pdSpeed, pdRadius, pdSteering, m_fMinTimeToCollision, numAgents );
	cutilCheckMsg( "SteerToAvoidNeighborsCUDAKernel failed." );
	CUDA_SAFE_CALL( cudaThreadSynchronize() );
}

void SteerToAvoidNeighborsCUDA::close(void )
{
	// Device data has changed. Instruct the VehicleGroup it needs to synchronize the host.
	m_pVehicleGroup->SetSyncHost();
}
