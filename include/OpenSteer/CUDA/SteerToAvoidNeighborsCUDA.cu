#include "SteerToAvoidNeighborsCUDA.cuh"

using namespace OpenSteer;

extern "C"
{
	__global__ void SteerToAvoidNeighborsCUDAKernel(	uint const*		pdKNNIndices,
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
}

SteerToAvoidNeighborsCUDA::SteerToAvoidNeighborsCUDA( VehicleGroup *pVehicleGroup, float const fMinTimeToCollision )
:	AbstractCUDAKernel( pVehicleGroup ),
	m_fMinTimeToCollision( fMinTimeToCollision )
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

	// Gather the required device pointers.
	uint const*		pdKNNIndices = m_pVehicleGroup->GetNearestNeighborData().pdKNNIndices();
	//float const*	pdKNNDistances = m_pVehicleGroup->GetNearestNeighborData().pdKNNDistances();

	float3 const*	pdPosition = m_pVehicleGroupData->pdPosition();
	float3 const*	pdDirection = m_pVehicleGroupData->pdForward();
	float3 const*	pdSide = m_pVehicleGroupData->pdSide();
	float const*	pdSpeed = m_pVehicleGroupData->pdSpeed();

	float const*	pdRadius = m_pVehicleGroupConst->pdRadius();

	float3 *		pdSteering = m_pVehicleGroupData->pdSteering();

	// Compute the size of shared memory needed for each block.
	size_t shMemSize = k * THREADSPERBLOCK * sizeof(uint);
	
	SteerToAvoidNeighborsCUDAKernel<<< grid, block, shMemSize >>>( pdKNNIndices, /*pdKNNDistances,*/ k, pdPosition, pdDirection, pdSide, pdSpeed, pdRadius, pdSteering, m_fMinTimeToCollision, getNumAgents() );
	cutilCheckMsg( "SteerToAvoidNeighborsCUDAKernel failed." );
	CUDA_SAFE_CALL( cudaThreadSynchronize() );
}

void SteerToAvoidNeighborsCUDA::close(void )
{
	// Device data has changed. Instruct the VehicleGroup it needs to synchronize the host.
	m_pVehicleGroup->SetSyncHost();
}
