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

	__global__ void KNNBruteForceCUDAKernelV2(	float3 const*	pdPosition,			// In:	Agent positions.
												uint *			pdKNNIndices,		// Out:	Indices of K Nearest Neighbors in pdPosition.
												float *			pdKNNDistances,		// Out:	Distances of each of the neighbors in pdKNNIndices.
												size_t const	k,					// In:	Number of neighbors to consider.
												size_t const	numAgents			// In:	Number of agents in the simulation.
												);

	__global__ void KNNBruteForceCUDAKernelV3(	float3 const*	pdPosition,			// Agent positions.
												uint *			pdKNNIndices,		// Output, indices of K Nearest Neighbors in pdPosition.
												float *			pdKNNDistances,		// Output, distances of the K Nearest Neighbors in pdPosition.
												size_t const	k,					// Number of neighbors to consider.
												size_t const	numAgents,			// Number of agents in the simulation.
												bool const		bSeed = false
											);
}

#include <thrust/device_ptr.h>
#include <thrust/sort.h>

using namespace OpenSteer;

#pragma region KNNBruteForceCUDA
KNNBruteForceCUDA::KNNBruteForceCUDA( VehicleGroup * pVehicleGroup )
:	AbstractCUDAKernel( pVehicleGroup, 1.f )
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
	float3 const*	pdPosition	= m_pVehicleGroupData->pdPosition();

	size_t const&	numAgents	= getNumAgents();
	size_t const&	k			= m_pVehicleGroup->GetNearestNeighborData().k();

	// Launch the KNNBruteForceCUDAKernel to compute distances to each other vehicle.
	KNNBruteForceCUDAKernel<<< grid, block >>>( pdPosition, m_pdDistanceMatrix, m_pdIndexMatrix, k, numAgents );
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
		thrust::sort_by_key( pdDistanceStart, pdDistanceEnd, pdIndexStart );
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
#pragma endregion


#pragma region KNNBruteForceCUDAV2
// 
//	V2 implementation.
//
KNNBruteForceCUDAV2::KNNBruteForceCUDAV2( VehicleGroup * pVehicleGroup )
:	AbstractCUDAKernel( pVehicleGroup, 1.f )
{
	m_pNearestNeighborData = &pVehicleGroup->GetNearestNeighborData();
}

void KNNBruteForceCUDAV2::init( void )
{
}

void KNNBruteForceCUDAV2::run( void )
{
	dim3 grid = gridDim();
	dim3 block = blockDim();

	// Gather required data.
	float3 const*	pdPosition = m_pVehicleGroupData->pdPosition();

	uint *			pdKNNIndices = m_pNearestNeighborData->pdKNNIndices();
	float *			pdKNNDistances = m_pNearestNeighborData->pdKNNDistances();
	uint			k = m_pNearestNeighborData->k();
	
	size_t			numAgents = getNumAgents();

	// Compute the size of shared memory needed for each block.
	size_t shMemSize = k * THREADSPERBLOCK * (sizeof(float) + sizeof(uint));

	KNNBruteForceCUDAKernelV2<<< grid, block, shMemSize >>>( pdPosition, pdKNNIndices, pdKNNDistances, k, numAgents );
	cutilCheckMsg( "KNNBruteForceCUDAKernelV2 failed." );
	CUDA_SAFE_CALL( cudaThreadSynchronize() );
}

void KNNBruteForceCUDAV2::close( void )
{
}
#pragma endregion


#pragma region KNNBruteForceCUDAV3
// 
//	V3 implementation.
//
KNNBruteForceCUDAV3::KNNBruteForceCUDAV3( VehicleGroup * pVehicleGroup )
:	AbstractCUDAKernel( pVehicleGroup, 1.f )
{
	m_pNearestNeighborData = &pVehicleGroup->GetNearestNeighborData();
}

void KNNBruteForceCUDAV3::init( void )
{
}

void KNNBruteForceCUDAV3::run( void )
{
	dim3 grid = gridDim();
	dim3 block = blockDim();

	// Gather required data.
	float3 const*	pdPosition = m_pVehicleGroupData->pdPosition();
	uint *			pdKNNIndices = m_pNearestNeighborData->pdKNNIndices();
	float *			pdKNNDistances = m_pNearestNeighborData->pdKNNDistances();
	size_t			numAgents = getNumAgents();
	uint			k = m_pNearestNeighborData->k();

	// Compute the size of shared memory needed for each block.
	size_t shMemSize = k * THREADSPERBLOCK * (sizeof(float) + sizeof(uint));

	// FIXME: there is a bug in the seeding part of KNNBruteForceV3
	KNNBruteForceCUDAKernelV3<<< grid, block, shMemSize >>>( pdPosition, pdKNNIndices, pdKNNDistances, k, numAgents, m_pNearestNeighborData->seedable() );
	//KNNBruteForceCUDAKernelV3<<< grid, block, shMemSize >>>( pdPosition, pdKNNIndices, pdKNNDistances, k, numAgents, false );
	cutilCheckMsg( "KNNBruteForceCUDAKernelV3 failed." );
	// Data will now be seedable.
	m_pNearestNeighborData->seedable( true );

	CUDA_SAFE_CALL( cudaThreadSynchronize() );
}

void KNNBruteForceCUDAV3::close( void )
{
}
#pragma endregion
