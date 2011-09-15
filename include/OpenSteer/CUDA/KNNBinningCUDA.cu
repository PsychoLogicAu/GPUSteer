#include "KNNBinningCUDA.cuh"

using namespace OpenSteer;

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
	__global__ void KNNBinningKernel(					// Group A
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

KNNBinningCUDA::KNNBinningCUDA( VehicleGroup * pVehicleGroup )
:	AbstractCUDAKernel( pVehicleGroup, 1.f )
{
	m_nCells = m_pVehicleGroup->GetBinData().getNumCells();
	m_pNearestNeighborData = &pVehicleGroup->GetNearestNeighborData();
}

void KNNBinningCUDA::init( void )
{
	// Bind the cell indices texture.
	KNNBinningCUDABindTexture( m_pVehicleGroup->GetBinData().pdCellIndexArray() );
}

void KNNBinningCUDA::run( void )
{
	dim3 grid = gridDim();
	dim3 block = blockDim();

	size_t const&	numAgents				= getNumAgents();
	uint const&		k						= m_pNearestNeighborData->k();
	float3 const&	worldSize				= m_pVehicleGroup->GetBinData().WorldSize();

	// Gather the required device pointers.
	float3 const*	pdPosition				= m_pVehicleGroupData->pdPosition();
	//float3 const*	pdDirection				= m_pVehicleGroupData->pdForward();
	//float const*	pdSpeed					= m_pVehicleGroupData->pdSpeed();

	// Pointers to output data.
	uint *			pdKNNIndices			= m_pNearestNeighborData->pdKNNIndices();
	float *			pdKNNDistances			= m_pNearestNeighborData->pdKNNDistances();

	uint *			pdCellIndices			= m_pNearestNeighborData->pdCellIndices();

	uint *			pdCellStart				= m_pVehicleGroup->GetBinData().pdCellStart();
	uint *			pdCellEnd				= m_pVehicleGroup->GetBinData().pdCellEnd();

	uint *			pdCellIndicesSorted		= m_pNearestNeighborData->pdCellIndicesSorted();
	uint *			pdAgentIndicesSorted	= m_pNearestNeighborData->pdAgentIndicesSorted();

	float3 *		pdPositionSorted		= m_pNearestNeighborData->pdPositionSorted();
	
	size_t const	radius					= m_pVehicleGroup->GetBinData().radius();
	size_t const	neighborsPerCell		= m_pVehicleGroup->GetBinData().neighborsPerCell();
	uint const*		pdCellNeighbors			= m_pVehicleGroup->GetBinData().pdCellNeighbors();

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

	// Build the database (get the bin indices for the agents).
	KNNBinningBuildDB<<< grid, block >>>( pdPosition, /*m_pdPositionNormalized,*/ pdAgentIndicesSorted, pdCellIndices, numAgents );
	cutilCheckMsg( "KNNBinningBuildDB failed." );
	CUDA_SAFE_CALL( cudaThreadSynchronize() );

	// Copy pdCellIndices to pdCellIndicesSorted.
	CUDA_SAFE_CALL( cudaMemcpy( pdCellIndicesSorted, pdCellIndices, numAgents * sizeof(uint), cudaMemcpyDeviceToDevice ) );

	// Sort m_pAgentIndices on m_pdCellIndicesSorted using thrust.
	thrust::sort_by_key(	thrust::device_ptr<uint>( pdCellIndicesSorted ),
							thrust::device_ptr<uint>( pdCellIndicesSorted + numAgents ),
							thrust::device_ptr<uint>( pdAgentIndicesSorted ) );

	// Set all cells to empty.
	CUDA_SAFE_CALL( cudaMemset( pdCellStart, 0xffffffff, m_nCells * sizeof(uint) ) );

	KNNBinningReorderDB<<< grid, block >>>(	pdPosition, /*m_pdPositionNormalized, pdDirection, pdSpeed,*/
											pdAgentIndicesSorted, pdCellIndicesSorted,
											pdPositionSorted, /*m_pdPositionNormalizedSorted, pdDirectionSorted, pdSpeedSorted,*/
											pdCellStart, pdCellEnd,
											numAgents
											);
	cutilCheckMsg( "KNNBinningReorderData failed" );
	CUDA_SAFE_CALL( cudaThreadSynchronize() );

	// Compute the size of shared memory needed for each block.
	size_t shMemSize = k * THREADSPERBLOCK * (sizeof(float) + sizeof(uint));

/*
	KNNBinningKernel<<< grid, block, shMemSize >>>(	pdPositionSorted,
													pdAgentIndicesSorted, pdCellIndicesSorted,
													pdCellStart, pdCellEnd,
													pdCellNeighbors, neighborsPerCell,
													pdKNNIndices, pdKNNDistances,
													k, radius, numAgents
											);
*/
	KNNBinningKernel<<< grid, block, shMemSize >>>(	pdPositionSorted, pdAgentIndicesSorted, pdCellIndicesSorted,
														pdPositionSorted, pdAgentIndicesSorted, pdCellIndicesSorted, 
														pdCellStart, pdCellEnd,
														pdCellNeighbors, neighborsPerCell, radius,
														pdKNNIndices, pdKNNDistances, k,
														numAgents, numAgents, true );
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
	sprintf_s( szString, "%f\n", elapsedTime );
	OutputDebugString( szString );

	// Destroy the events.
	cudaEventDestroy( start );
	cudaEventDestroy( stop );
#endif
}

void KNNBinningCUDA::close( void )
{
	// Unbind the texture.
	KNNBinningCUDAUnbindTexture();
}