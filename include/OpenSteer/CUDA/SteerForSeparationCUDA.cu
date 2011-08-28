#include "SteerForSeparationCUDA.cuh"

using namespace OpenSteer;

extern "C"
{
	__global__ void SteerForSeparationKernel(	uint const*		pdKNNIndices,
												size_t const	k,
												
												float3 const*	pdPosition,
		
												float3 *		pdSteering,
												float const		weight,
												size_t const	numAgents
												);
}

SteerForSeparationCUDA::SteerForSeparationCUDA(	VehicleGroup * pVehicleGroup, float const fWeight )
:	AbstractCUDAKernel( pVehicleGroup, fWeight )
{
	// Nothing to do.
}

void SteerForSeparationCUDA::init( void )
{
	// Nothing to do.
}

void SteerForSeparationCUDA::run( void )
{
	dim3 grid = gridDim();
	dim3 block = blockDim();

	size_t const	numAgents = getNumAgents();
	size_t const	k = m_pVehicleGroup->GetNearestNeighborData().k();

	// Gather required device pointers.
	float3 const*	pdPosition = m_pVehicleGroupData->pdPosition();
	float3 *		pdSteering = m_pVehicleGroupData->pdSteering();
	uint const*		pdKNNIndices = m_pVehicleGroup->GetNearestNeighborData().pdKNNIndices();

	// Compute the size of shared memory needed for each block.
	size_t shMemSize = k * THREADSPERBLOCK * sizeof(uint);

	// Launch the kernel.
	SteerForSeparationKernel<<< grid, block, shMemSize >>>( pdKNNIndices, k, pdPosition, pdSteering, getWeight(), numAgents );
	cutilCheckMsg( "SteerForSeparationKernel failed." );
	CUDA_SAFE_CALL( cudaThreadSynchronize() );
}

void SteerForSeparationCUDA::close( void )
{
	// Nothing to do.
}
