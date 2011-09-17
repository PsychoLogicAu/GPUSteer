#include "KNNBinningCUDA.cuh"

using namespace OpenSteer;

#include "KNNBinData.cuh"

#include <thrust/sort.h>

#include "DebugUtils.h"

#define TIMING

// Kernel file function prototypes.
extern "C"
{
	// Bind texCellIndices to the cudaArray.
	__host__ void KNNBinningCUDABindTexture( cudaArray * pCudaArray );
	__host__ void KNNBinningCUDAUnbindTexture( void );

	__global__ void KNNBinningBuildDB(		float3 const*	pdPosition,				// In:	Positions of each agent.
											size_t *		pdAgentIndices,			// Out:	Indices of each agent.
											size_t *		pdCellIndices,			// Out:	Indices of the cell each agent is in.
											size_t const	numAgents
											);

	// Kernel to sort position/direction/speed based on pdAgentIndices, and to compute start and end indices of cells.
	__global__ void KNNBinningReorderDB(	float3 const*	pdPosition,			// In: Agent positions.
								
											uint const*		pdAgentIndices,		// In: (sorted) agent index.
											uint const*		pdCellIndices,		// In: (sorted) cell index agent is in.

											float3 *		pdPositionSorted,	// Out: Sorted agent positions.

											uint *			pdCellStart,		// Out: Start index of this cell in pdCellIndices.
											uint *			pdCellEnd,			// Out: End index of this cell in pdCellIndices.

											size_t const	numAgents
											);
/*
	__global__ void KNNBinningKernel(		float3 const*	pdPositionSorted,			// In:	(sorted) Agent positions.

											uint const*		pdAgentIndices,				// In:	(sorted) Indices of each agent.
											uint const*		pdCellIndices,				// In:	(sorted) Indices of the cell each agent is currently in.

											uint const*		pdCellStart,				// In:	Start index of each cell in pdCellIndices.
											uint const*		pdCellEnd,					// In:	End index of each cell in pdCellIndices.

											uint const*		pdCellNeighbors,			// In:	Indices of the neighbors to radius distance of each cell.
											size_t const	neighborsPerCell,			// In:	Number of neighbors per cell in the pdCellNeighbors array.

											uint *			pdKNNIndices,				// Out:	Indices of K Nearest Neighbors in pdPosition.
											float *			pdKNNDistances,				// Out:	Distances of the K Nearest Neighbors in pdPosition.

											size_t const	k,							// In:	Number of neighbors to consider.
											size_t const	radius,						// In:	Maximum radius (in cells) to consider.
											size_t const	numAgents					// In:	Number of agents in the simulation.
											);
*/
	__global__ void KNNBinningKernel(		// Group A
											float3 const*	pdAPositionSorted,			// In:	Sorted group A positions.

											uint const*		pdAIndices,					// In:	Sorted group A indices
											uint const*		pdACellIndices,				// In:	Sorted group A cell indices.

											// Group B
											float3 const*	pdBPositionSorted,			// In:	Sorted group B positions.

											uint const*		pdBIndices,					// In:	Sorted group B indices
											uint const*		pdBCellIndices,				// In:	Sorted group B cell indices.

											uint const*		pdBCellStart,				// In:	Start index of each cell in pdBCellIndices.
											uint const*		pdBCellEnd,					// In:	End index of each cell in pdBCellIndices.

											// Cell neighbor info.
											uint const*		pdCellNeighbors,			// In:	Indices of the neighbors to radius distance of each cell.
											size_t const	neighborsPerCell,			// In:	Number of neighbors per cell in the pdCellNeighbors array.
											size_t const	radius,						// In:	Search radius (in cells) to consider.

											// Output data.
											uint *			pdKNNIndices,				// Out:	Indices of K Nearest Neighbors in pdPosition.
											float *			pdKNNDistances,				// Out:	Distances of the K Nearest Neighbors in pdPosition.

											size_t const	k,							// In:	Number of neighbors to consider.
											size_t const	numA,						// In:	Size of group A.
											size_t const	numB,						// In:	Size of group B.
											bool const		groupWithSelf				// In:	Are we testing this group with itself? (group A == group B)
											);
}

#pragma region KNNBinningUpdateDBCUDA

KNNBinningUpdateDBCUDA::KNNBinningUpdateDBCUDA( AgentGroup * pAgentGroup, KNNBinData * pKNNBinData )
:	AbstractCUDAKernel( pAgentGroup, 1.f ),
	m_pKNNBinData( pKNNBinData )
{
	// Nothing to do.
}


void KNNBinningUpdateDBCUDA::init( void )
{
	// Bind the lookup texture.
	KNNBinningCUDABindTexture( m_pKNNBinData->pdCellIndexArray() );
}

void KNNBinningUpdateDBCUDA::run( void )
{
	dim3 grid = gridDim();
	dim3 block = blockDim();

	// Gather required data.
	float3 const*	pdPosition				= m_pAgentGroupData->pdPosition();
	
	uint *			pdCellIndices			= m_pAgentGroup->GetKNNDatabase().pdCellIndices();

	uint *			pdAgentIndicesSorted	= m_pAgentGroup->GetKNNDatabase().pdAgentIndicesSorted();
	uint *			pdCellIndicesSorted		= m_pAgentGroup->GetKNNDatabase().pdCellIndicesSorted();

	float3 *		pdPositionSorted		= m_pAgentGroup->GetKNNDatabase().pdPositionSorted();
	uint *			pdCellStart				= m_pAgentGroup->GetKNNDatabase().pdCellStart();
	uint *			pdCellEnd				= m_pAgentGroup->GetKNNDatabase().pdCellEnd();

	uint const&		numAgents				= getNumAgents();

#if defined TIMING
	//
	//	TIMING: hard to get exact times with profiling, too many operations.
	//
	// Events for timing the complete operation.
	cudaEvent_t start, stop;
	cudaEventCreate( &start );
	cudaEventCreate( &stop );
	cudaEventRecord( start, 0 );
#endif

	// Call KNNBinningBuildDB to build the database. 
	KNNBinningBuildDB<<< grid, block >>>( pdPosition, pdAgentIndicesSorted, pdCellIndices, numAgents );
	cutilCheckMsg( "KNNBinningBuildDB failed." );
	CUDA_SAFE_CALL( cudaThreadSynchronize() );

	// Copy pdCellIndices to  pdCellIndicesSorted.
	CUDA_SAFE_CALL( cudaMemcpy( pdCellIndicesSorted, pdCellIndices, numAgents * sizeof(uint), cudaMemcpyDeviceToDevice ) );

	// Sort pdAgentIndicesSorted on pdCellIndicesSorted using thrust.
	thrust::sort_by_key(	thrust::device_ptr<uint>( pdCellIndicesSorted ),
							thrust::device_ptr<uint>( pdCellIndicesSorted + numAgents ),
							thrust::device_ptr<uint>( pdAgentIndicesSorted ) );

	// Set all cells to empty.
	CUDA_SAFE_CALL( cudaMemset( pdCellStart, 0xffffffff, m_pKNNBinData->getNumCells() * sizeof(uint) ) );

	// Call KNNBinningReorderDB to re-order the data in the DB.
	KNNBinningReorderDB<<< grid, block >>>( pdPosition, pdAgentIndicesSorted, pdCellIndicesSorted, pdPositionSorted, pdCellStart, pdCellEnd, numAgents );
	cutilCheckMsg( "KNNBinningReorderDB failed." );
	CUDA_SAFE_CALL( cudaThreadSynchronize() );

#if defined TIMING
	//
	//	TIMING:
	//
	cudaEventRecord( stop, 0 );
	cudaEventSynchronize( stop );
	
	float elapsedTime;
	cudaEventElapsedTime( &elapsedTime, start, stop );
	char szString[128] = {0};
	sprintf_s( szString, "KNNBinningUpdateDBCUDA,%f\n", elapsedTime );
	OutputDebugStringToFile( szString );

	// Destroy the events.
	cudaEventDestroy( start );
	cudaEventDestroy( stop );
#endif
}

void KNNBinningUpdateDBCUDA::close( void )
{
	// Unbind the texture.
	KNNBinningCUDAUnbindTexture();

	// The AgentGroup's database has now changed.
	m_pAgentGroup->SetSyncHost();
}
#pragma endregion


#pragma region KNNBinningCUDA

KNNBinningCUDA::KNNBinningCUDA( AgentGroup * pAgentGroup, KNNData * pKNNData, KNNBinData * pKNNBinData, BaseGroup * pOtherGroup )
:	AbstractCUDAKernel( pAgentGroup, 1.f ),
	m_pKNNData( pKNNData ),
	m_pKNNBinData( pKNNBinData ),
	m_pOtherGroup( pOtherGroup )
{
}

void KNNBinningCUDA::init( void )
{
	// Bind the cell indices texture.
	//KNNBinningCUDABindTexture( m_pKNNBinData->pdCellIndexArray() );
}

void KNNBinningCUDA::run( void )
{
	dim3 grid = gridDim();
	dim3 block = blockDim();

	// Gather the required data.
	float3 const*		pdAPositionSorted		= m_pAgentGroupData->pdPosition();
	uint const*			pdAIndices				= m_pAgentGroup->GetKNNDatabase().pdAgentIndicesSorted();
	uint const*			pdACellIndices			= m_pAgentGroup->GetKNNDatabase().pdCellIndicesSorted();

	float3 const*		pdBPositionSorted		= m_pOtherGroup->pdPosition();
	uint const*			pdBIndices				= m_pOtherGroup->GetKNNDatabase().pdAgentIndicesSorted();
	uint const*			pdBCellIndices			= m_pOtherGroup->GetKNNDatabase().pdCellIndicesSorted();

	uint const*			pdBCellStart			= m_pOtherGroup->GetKNNDatabase().pdCellStart();
	uint const*			pdBCellEnd				= m_pOtherGroup->GetKNNDatabase().pdCellEnd();

	uint const*			pdCellNeighbors			= m_pKNNBinData->pdCellNeighbors();
	uint const&			neighborsPerCell		= m_pKNNBinData->neighborsPerCell();
	uint const&			radius					= m_pKNNBinData->radius();

	uint *				pdKNNIndices			= m_pKNNData->pdKNNIndices();
	float *				pdKNNDistances			= m_pKNNData->pdKNNDistances();

	uint const&			k						= m_pOtherGroup->GetKNNDatabase().k();
	uint const&			numA					= getNumAgents();
	uint const&			numB					= m_pOtherGroup->Size();

	bool const			groupWithSelf			= m_pAgentGroup == m_pOtherGroup;

	// Compute the size of shared memory needed for each block.
	size_t shMemSize = k * THREADSPERBLOCK * (sizeof(float) + sizeof(uint));

#if defined TIMING
	//
	//	TIMING: hard to get exact times with profiling, too many operations.
	//
	// Events for timing the complete operation.
	cudaEvent_t start, stop;
	cudaEventCreate( &start );
	cudaEventCreate( &stop );
	cudaEventRecord( start, 0 );
#endif

	// Call the KNNBinning kernel.
	KNNBinningKernel<<< grid, block, shMemSize >>>(	pdAPositionSorted,
													pdAIndices, pdACellIndices, 
													pdBPositionSorted, pdBIndices, pdBCellIndices, pdBCellStart, pdBCellEnd,
													pdCellNeighbors, neighborsPerCell, radius,
													pdKNNIndices, pdKNNDistances, k,
													numA, numB, groupWithSelf
													);
	cutilCheckMsg( "KNNBinningKernel failed." );
	CUDA_SAFE_CALL( cudaThreadSynchronize() );

#if defined TIMING
	//
	//	TIMING:
	//
	cudaEventRecord( stop, 0 );
	cudaEventSynchronize( stop );
	
	float elapsedTime;
	cudaEventElapsedTime( &elapsedTime, start, stop );
	char szString[128] = {0};
	sprintf_s( szString, "KNNBinningCUDA,%f\n", elapsedTime );
	OutputDebugStringToFile( szString );

	// Destroy the events.
	cudaEventDestroy( start );
	cudaEventDestroy( stop );
#endif
}

void KNNBinningCUDA::close( void )
{
	// Unbind the texture.
	//KNNBinningCUDAUnbindTexture();

	// The KNNData has most likely changed.
	m_pKNNData->setSyncHost();
}

#pragma endregion
