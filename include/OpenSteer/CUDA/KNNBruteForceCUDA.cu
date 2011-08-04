#include "KNNBruteForceCUDA.cuh"

extern "C" __global__ void KNNBruteForceCUDAKernel(	float3 const*	pdPosition,			// Agent positions.
											float const*	pdDistanceMatrix,	// Global storage for distance matrix.
											size_t const*	pdIndexMatrix,		// The indices which match postions in pdDistanceMatrix.
											//size_t *		pdKNNIndices,		// Output, indices of K Nearest Neighbors in pdPosition.
											size_t const	k,					// Number of neighbors to consider.
											size_t const	numAgents			// Number of agents in the simulation.
										);

#define USE_THRUST

#ifdef USE_THRUST
#include <thrust/sort.h>
#endif

using namespace OpenSteer;

KNNBruteForceCUDA::KNNBruteForceCUDA( VehicleGroup * pVehicleGroup, size_t const k )
:	AbstractCUDAKernel( pVehicleGroup ),
	m_k( k )
{ }

void KNNBruteForceCUDA::init( void )
{
	size_t numAgents = getNumAgents();
	size_t numAgentsSquared = numAgents * numAgents;
	
	// Allocate the temporary device storage.
	cudaMalloc( &m_pdDistanceMatrix, numAgentsSquared * sizeof(float) );
	cudaMalloc( &m_pdIndexMatrix, numAgentsSquared * sizeof(size_t) );

}

void KNNBruteForceCUDA::run( void )
{
	dim3 grid = gridDim();
	dim3 block = blockDim();

	// Gather the required device pointers.
	float3 const*	pdPosition = m_pdVehicleGroupData->dpPosition();

	size_t numAgents = getNumAgents();

	// Launch the KNNBruteForceCUDAKernel to compute distances to each other vehicle.
	KNNBruteForceCUDAKernel<<< grid, block >>>( pdPosition, m_pdDistanceMatrix, m_pdIndexMatrix, m_k, numAgents );

	// For each agent...
	for( size_t i = 0; i < numAgents; i++ )
	{
		// Pointers to the matrix row start and end for this agent (keys).
		float * pdPositionStart = m_pdDistanceMatrix + (i * numAgents);
		float * pdPositionEnd = pdPositionStart + (numAgents);
		// Pointer to the index matrix row for this agent (values).
		size_t * pdIndexBase = m_pdIndexMatrix + (i * numAgents);

		// Sort the results (using thrust)
#ifdef USE_THRUST
		thrust::sort_by_key( pdPositionStart, pdPositionEnd, pdIndexBase );
#endif

	}
}

void KNNBruteForceCUDA::close( void )
{
	// Deallocate the temporary device storage.
	cudaFree( m_pdDistanceMatrix );
	cudaFree( m_pdIndexMatrix );
}
