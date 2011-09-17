#include "KNNBruteForceCUDA.cuh"

#include <iostream>
#include <fstream>

extern "C"
{
	__global__ void KNNBruteForceCUDAKernel(	float3 const*	pdPosition,			// Agent positions.
												float *			pdDistanceMatrix,	// Global storage for distance matrix.
												size_t *		pdIndexMatrix,		// The indices which match postions in pdDistanceMatrix.
												size_t const	k,					// Number of neighbors to consider.
												size_t const	numAgents,			// Number of agents in the simulation.
												float3 const*	pdPositionOther,
												uint const		numOther,
												bool const		groupWithSelf
												);

	__global__ void KNNBruteForceCUDAKernelV2(	float3 const*	pdPosition,			// In:	Agent positions.
												uint *			pdKNNIndices,		// Out:	Indices of K Nearest Neighbors in pdPosition.
												float *			pdKNNDistances,		// Out:	Distances of each of the neighbors in pdKNNIndices.
												size_t const	k,					// In:	Number of neighbors to consider.
												size_t const	numAgents,			// In:	Number of agents in the simulation.
												float3 const*	pdPositionOther,
												uint const		numOther,
												bool const		groupWithSelf
												);

	__global__ void KNNBruteForceCUDAKernelV3(	float3 const*	pdPosition,			// Agent positions.

												uint *			pdKNNIndices,		// Output, indices of K Nearest Neighbors in pdPosition.
												float *			pdKNNDistances,		// Output, distances of the K Nearest Neighbors in pdPosition.

												size_t const	k,					// Number of neighbors to consider.
												size_t const	numAgents,			// Number of agents in the simulation.

												float3 const*	pdPositionOther,
												uint const		numOther,
												bool const		groupWithSelf,

												bool const		bSeed = false
												);
}

#include <thrust/device_ptr.h>
#include <thrust/sort.h>

using namespace OpenSteer;

#pragma region KNNBruteForceCUDA
KNNBruteForceCUDA::KNNBruteForceCUDA( AgentGroup * pAgentGroup, KNNData * pKNNData, BaseGroup * pOtherGroup )
:	AbstractCUDAKernel( pAgentGroup, 1.f ),
	m_pKNNData( pKNNData ),
	m_pOtherGroup( pOtherGroup )
{ }

void KNNBruteForceCUDA::init( void )
{
	size_t const& numAgents = getNumAgents();
	size_t const numAgentsSquared = numAgents * numAgents;

	// Allocate the temporary device storage.
	CUDA_SAFE_CALL( cudaMalloc( &m_pdDistanceMatrix, numAgentsSquared * sizeof(float) ) );
	CUDA_SAFE_CALL( cudaMalloc( &m_pdIndexMatrix, numAgentsSquared * sizeof(uint) ) );
}

void KNNBruteForceCUDA::run( void )
{
	dim3 grid = gridDim();
	dim3 block = blockDim();

	// Gather the required device pointers.
	float3 const*	pdPosition		= m_pAgentGroupData->pdPosition();

	uint const&		numAgents		= getNumAgents();
	uint const&		k				= m_pKNNData->k();

	uint const&		numOther		= m_pOtherGroup->Size();
	float3 const*	pdPositionOther = m_pOtherGroup->pdPosition();
	bool const		groupWithSelf	= pdPosition == pdPositionOther;

#if defined TIMING
	// Events for timing the sort operations.
	cudaEvent_t start, stop;
	cudaEventCreate( &start );
	cudaEventCreate( &stop );
	cudaEventRecord( start, 0 );
#endif

	// Launch the KNNBruteForceCUDAKernel to compute distances to each other vehicle.
	KNNBruteForceCUDAKernel<<< grid, block >>>( pdPosition, m_pdDistanceMatrix, m_pdIndexMatrix, k, numAgents, pdPositionOther, numOther, groupWithSelf );
	cutilCheckMsg( "KNNBruteForceCUDAKernel failed." );
	CUDA_SAFE_CALL( cudaThreadSynchronize() );

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

		// Copy the first k elements to the KNNData structure for output.
		CUDA_SAFE_CALL( cudaMemcpy( m_pKNNData->pdKNNDistances() + i*k, m_pdDistanceMatrix, k * sizeof(float), cudaMemcpyDeviceToDevice ) );
		CUDA_SAFE_CALL( cudaMemcpy( m_pKNNData->pdKNNIndices() + i*k, m_pdIndexMatrix, k * sizeof(uint), cudaMemcpyDeviceToDevice ) );
	}
	
#if defined TIMING
	cudaEventRecord( stop, 0 );
	cudaEventSynchronize( stop );
	
	float elapsedTime;
	cudaEventElapsedTime( &elapsedTime, start, stop );
	char szString[128] = {0};
	sprintf_s( szString, "KNNBruteForceCUDA,%f\n", elapsedTime );
	OutputDebugStringToFile( szString );

	// Destroy the events.
	cudaEventDestroy( start );
	cudaEventDestroy( stop );
#endif
}

void KNNBruteForceCUDA::close( void )
{
	// Deallocate the temporary device storage.
	CUDA_SAFE_CALL( cudaFree( m_pdDistanceMatrix ) );
	CUDA_SAFE_CALL( cudaFree( m_pdIndexMatrix ) );
}
#pragma endregion


#pragma region KNNBruteForceCUDAV2
// 
//	V2 implementation.
//
KNNBruteForceCUDAV2::KNNBruteForceCUDAV2( AgentGroup * pAgentGroup, KNNData * pKNNData, BaseGroup * pOtherGroup )
:	AbstractCUDAKernel( pAgentGroup, 1.f ),
	m_pKNNData( pKNNData ),
	m_pOtherGroup( pOtherGroup )
{
}

void KNNBruteForceCUDAV2::init( void )
{
}

void KNNBruteForceCUDAV2::run( void )
{
	dim3 grid = gridDim();
	dim3 block = blockDim();

	// Gather required data.
	float3 const*	pdPosition		= m_pAgentGroupData->pdPosition();

	uint *			pdKNNIndices	= m_pKNNData->pdKNNIndices();
	float *			pdKNNDistances	= m_pKNNData->pdKNNDistances();
	uint const&		k				= m_pKNNData->k();
	
	uint const&		numAgents		= getNumAgents();

	float3 const*	pdPositionOther	= m_pOtherGroup->pdPosition();
	uint const&		numOther		= m_pOtherGroup->Size();
	bool const		groupWithSelf	= pdPosition == pdPositionOther;

	// Compute the size of shared memory needed for each block.
	size_t shMemSize = k * THREADSPERBLOCK * (sizeof(float) + sizeof(uint));

	KNNBruteForceCUDAKernelV2<<< grid, block, shMemSize >>>( pdPosition, pdKNNIndices, pdKNNDistances, k, numAgents, pdPositionOther, numOther, groupWithSelf );
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
KNNBruteForceCUDAV3::KNNBruteForceCUDAV3( AgentGroup * pAgentGroup, KNNData * pKNNData, BaseGroup * pOtherGroup )
:	AbstractCUDAKernel( pAgentGroup, 1.f ),
	m_pKNNData( pKNNData ),
	m_pOtherGroup( pOtherGroup )
{
}

void KNNBruteForceCUDAV3::init( void )
{
	// Nothing to do.
}

void KNNBruteForceCUDAV3::run( void )
{
	dim3 grid = gridDim();
	dim3 block = blockDim();

	// Gather required data.
	float3 const*	pdPosition		= m_pAgentGroupData->pdPosition();
	uint const&		numAgents		= getNumAgents();

	uint *			pdKNNIndices	= m_pKNNData->pdKNNIndices();
	float *			pdKNNDistances	= m_pKNNData->pdKNNDistances();
	uint const&		k				= m_pKNNData->k();
	bool const&		seedable		= m_pKNNData->seedable();

	float3 const*	pdPositionOther	= m_pOtherGroup->pdPosition();
	uint const&		numOther		= m_pOtherGroup->Size();
	bool const		groupWithSelf	= m_pAgentGroup == m_pOtherGroup;

	// Compute the size of shared memory needed for each block.
	size_t shMemSize = k * THREADSPERBLOCK * (sizeof(float) + sizeof(uint));

	// FIXME: there is a bug in the seeding part of KNNBruteForceV3
	KNNBruteForceCUDAKernelV3<<< grid, block, shMemSize >>>( pdPosition, pdKNNIndices, pdKNNDistances, k, numAgents, pdPositionOther, numOther, groupWithSelf, seedable );
	cutilCheckMsg( "KNNBruteForceCUDAKernelV3 failed." );

	// Data will now be seedable.
	m_pKNNData->seedable( true );

	CUDA_SAFE_CALL( cudaThreadSynchronize() );
}

void KNNBruteForceCUDAV3::close( void )
{
	// The KNNData has most likely changed.
	m_pKNNData->setSyncHost();
}
#pragma endregion
