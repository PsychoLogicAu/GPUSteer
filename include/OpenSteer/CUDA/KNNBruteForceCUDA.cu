#include "KNNBruteForceCUDA.cuh"

#include <iostream>
#include <fstream>

extern "C"
{
	__global__ void KNNBruteForceCUDAKernel(	float3 const*	pdPosition,			// Agent positions.
												float const*	pdDistanceMatrix,	// Global storage for distance matrix.
												size_t const*	pdIndexMatrix,		// The indices which match postions in pdDistanceMatrix.
												//size_t *		pdKNNIndices,		// Output, indices of K Nearest Neighbors in pdPosition.
												size_t const	k,					// Number of neighbors to consider.
												size_t const	numAgents			// Number of agents in the simulation.
											);

	__global__ void KNNBruteForceCUDAKernelV2(	float3 const*	pdPosition,			// Agent positions.
												uint *			pdKNNIndices,		// Output, indices of K Nearest Neighbors in pdPosition.
												size_t const	k,					// Number of neighbors to consider.
												size_t const	numAgents			// Number of agents in the simulation.
											);
}

#define USE_THRUST

#ifdef USE_THRUST
#include <thrust/device_ptr.h>
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
	cutilCheckMsg( "cudaMalloc failed." );
	cudaMalloc( &m_pdIndexMatrix, numAgentsSquared * sizeof(size_t) );
	cutilCheckMsg( "cudaMalloc failed." );
}

void KNNBruteForceCUDA::run( void )
{
	dim3 grid = gridDim();
	dim3 block = blockDim();

	// Gather the required device pointers.
	float3 const*	pdPosition = m_pVehicleGroupData->pdPosition();

	size_t numAgents = getNumAgents();

	// Launch the KNNBruteForceCUDAKernel to compute distances to each other vehicle.
	KNNBruteForceCUDAKernel<<< grid, block >>>( pdPosition, m_pdDistanceMatrix, m_pdIndexMatrix, m_k, numAgents );
	cutilCheckMsg( "KNNBruteForceCUDAKernel failed." );

	CUDA_SAFE_CALL( cudaThreadSynchronize() );

	// Events for timing the sort operations.
	cudaEvent_t start, stop;
	cudaEventCreate( &start );
	cudaEventCreate( &stop );
	cudaEventRecord( start, 0 );


	// For each agent...
	for( size_t i = 0; i < numAgents; i++ )
	{
		// Pointers to the matrix row start and end for this agent (keys).
		//float * pdPositionStart = m_pdDistanceMatrix + (i * numAgents);
		thrust::device_ptr< float > pdDistanceStart( m_pdDistanceMatrix + (i * numAgents) );
		//float * pdPositionEnd = pdPositionStart + (numAgents);
		thrust::device_ptr< float > pdDistanceEnd( m_pdDistanceMatrix + (i * numAgents) );
		// Pointer to the index matrix row for this agent (values).
		//size_t * pdIndexBase = m_pdIndexMatrix + (i * numAgents);
		thrust::device_ptr< size_t > pdIndexStart( m_pdIndexMatrix + (i * numAgents) );

		// Sort the results (using thrust)
#ifdef USE_THRUST
		thrust::sort_by_key( pdDistanceStart, pdDistanceEnd, pdIndexStart );
#endif

	}

	
	cudaEventRecord( stop, 0 );
	cudaEventSynchronize( stop );
	
	float elapsedTime;
	cudaEventElapsedTime( &elapsedTime, start, stop );
	char szString[128] = {0};
	sprintf_s( szString, "%f\n", elapsedTime );
	OutputDebugString( szString );


	// Destroy the events.
	cudaEventDestroy( start );
	cudaEventDestroy( stop );
}

void KNNBruteForceCUDA::close( void )
{
	// Deallocate the temporary device storage.
	cudaFree( m_pdDistanceMatrix );
	cudaFree( m_pdIndexMatrix );
}


// 
//	V2 implementation.
//
KNNBruteForceCUDAV2::KNNBruteForceCUDAV2( VehicleGroup * pVehicleGroup, size_t const k )
:	AbstractCUDAKernel( pVehicleGroup ),
	m_k( k )
{
}

void KNNBruteForceCUDAV2::init( void )
{
	// Allocate m_pdKNNIndices...
	CUDA_SAFE_CALL( cudaMalloc( &m_pdKNNIndices, getNumAgents() * m_k * sizeof(uint) ) );
}

void KNNBruteForceCUDAV2::run( void )
{
	dim3 grid = gridDim();
	dim3 block = blockDim();

	// Gather required data.
	float3 const*	pdPosition = m_pVehicleGroupData->pdPosition();
	size_t			numAgents = getNumAgents();

	// Compute the size of shared memory needed for each block.
	size_t shMemSize = 2 * m_k * THREADSPERBLOCK * sizeof(float);

	KNNBruteForceCUDAKernelV2<<< grid, block, shMemSize >>>( pdPosition, m_pdKNNIndices, m_k, numAgents );
	cutilCheckMsg( "KNNBruteForceCUDAKernelV2 failed." );

	CUDA_SAFE_CALL( cudaThreadSynchronize() );
}

void KNNBruteForceCUDAV2::close( void )
{
	// Free m_pdKNNIndices.
	cudaFree( m_pdKNNIndices );
}
