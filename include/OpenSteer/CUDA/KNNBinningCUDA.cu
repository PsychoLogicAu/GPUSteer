#include "KNNBinningCUDA.cuh"

using namespace OpenSteer;

#include <thrust/sort.h>

#include "DebugUtils.h"

// Kernel file function prototypes.
extern "C"
{
	// Bind texCellIndices to the cudaArray.
	__host__ void KNNBinningCUDABindTexture( cudaArray * pCudaArray );
	__host__ void KNNBinningCUDAUnbindTexture( void );

	// Kernel to set initial bin indices of vehicles in the simulation.
	__global__ void KNNBinningBuildDB(	float3 const*	pdPosition,				// In:	Positions of each agent.
										size_t *		pdAgentIndices,			// Out:	Indices of each agent.
										size_t *		pdCellIndices,			// Out:	Indices of the cell each agent is in.
										size_t const	numAgents/*,				// In:	Number of agents in the simulation.
										float3 const	worldSize				// In:	Extents of the world (for normalizing the positions).*/
										);

	// Kernel to sort position/direction/speed based on pdAgentIndices, and to compute start and end indices of cells.
	__global__ void KNNBinningReorderData(	float3 const*	pdPosition,			// In: Agent positions.
											float3 const*	pdDirection,		// In: Agent directions.
											float const*	pdSpeed,			// In: Agent speeds.
					
											uint const*		pdAgentIndices,		// In: (sorted) agent index.
											uint const*		pdCellIndices,		// In: (sorted) cell index agent is in.

											float3 *		pdPositionSorted,	// Out: Sorted agent positions.
											float3 *		pdDirectionSorted,	// Out: Sorted agent directions.
											float *			pdSpeedSorted,		// Out: Sorted agent speeds.

											uint *			pdCellStart,		// Out: Start index of this cell in pdCellIndices.
											uint *			pdCellEnd,			// Out: End index of this cell in pdCellIndices.

											size_t const	numAgents
											);

	__global__ void KNNBinningKernel(	float3 const*	pdPositionSorted,	// In:	(sorted) Agent positions.

										uint const*		pdAgentIndices,		// In:	(sorted) Indices of each agent.
										uint const*		pdCellIndices,		// In:	(sorted) Indices of the cell each agent is in.
									
										uint const*		pdCellStart,		// In:	Start index of each cell in pdCellIndices.
										uint const*		pdCellEnd,			// In:	End index of each cell in pdCellIndices.

										uint *			pdKNNIndices,		// Out:	Indices of K Nearest Neighbors in pdPosition.
										float *			pdKNNDistances,		// Out:	Distances of the K Nearest Neighbors in pdPosition.

										size_t const	k,					// In:	Number of neighbors to consider.
										size_t const	radius,				// In:	Maximum radius (in cells) to consider.
										size_t const	numAgents			// In:	Number of agents in the simulation.
										);
}

KNNBinningCUDA::KNNBinningCUDA( VehicleGroup * pVehicleGroup )
:	AbstractCUDAKernel( pVehicleGroup )
{
	m_nCells = m_pVehicleGroup->GetBinData().getNumCells();
	m_pNearestNeighborData = &pVehicleGroup->GetNearestNeighborData();
}

void KNNBinningCUDA::init( void )
{
	// Bind the cell indices texture.
	KNNBinningCUDABindTexture( m_pVehicleGroup->GetBinData().pdCellIndexArray() );

	CUDA_SAFE_CALL( cudaMalloc( &m_pdCellStart, m_nCells * sizeof(uint) ) );
	CUDA_SAFE_CALL( cudaMalloc( &m_pdCellEnd, m_nCells * sizeof(uint) ) );
}

void KNNBinningCUDA::run( void )
{
	dim3 grid = gridDim();
	dim3 block = blockDim();

	size_t const&	numAgents = getNumAgents();
	uint const&		k = m_pNearestNeighborData->k();
	float3 const&	worldSize = m_pVehicleGroup->GetBinData().WorldSize();

	// Gather the required device pointers.
	float3 const*	pdPosition = m_pVehicleGroupData->pdPosition();
	float3 const*	pdDirection = m_pVehicleGroupData->pdForward();
	float const*	pdSpeed = m_pVehicleGroupData->pdSpeed();

	// Pointers to output data.
	uint *			pdKNNIndices = m_pNearestNeighborData->pdKNNIndices();
	float *			pdKNNDistances = m_pNearestNeighborData->pdKNNDistances();

	uint *			pdCellIndices = m_pNearestNeighborData->pdCellIndices();

	uint *			pdCellIndicesSorted = m_pNearestNeighborData->pdCellIndicesSorted();
	uint *			pdAgentIndicesSorted = m_pNearestNeighborData->pdAgentIndicesSorted();

	float3 *		pdPositionSorted = m_pNearestNeighborData->pdPositionSorted();
	float3 *		pdDirectionSorted = m_pNearestNeighborData->pdDirectionSorted();
	float *			pdSpeedSorted = m_pNearestNeighborData->pdSpeedSorted();

	//
	//	TIMING: hard to get exact times with profiling, too many operations.
	//
	// Events for timing the complete operation.
	cudaEvent_t start, stop;
	cudaEventCreate( &start );
	cudaEventCreate( &stop );
	cudaEventRecord( start, 0 );

	// Build the database (get the bin indices for the agents).
	KNNBinningBuildDB<<< grid, block >>>( pdPosition, pdAgentIndicesSorted, pdCellIndices, numAgents );
	cutilCheckMsg( "KNNBinningBuildDB failed." );
	CUDA_SAFE_CALL( cudaThreadSynchronize() );

	// Copy pdCellIndices to pdCellIndicesSorted.
	CUDA_SAFE_CALL( cudaMemcpy( pdCellIndicesSorted, pdCellIndices, numAgents * sizeof(uint), cudaMemcpyDeviceToDevice ) );

	// Sort m_pAgentIndices on m_pdCellIndicesSorted using thrust.
	thrust::sort_by_key(	thrust::device_ptr<uint>( pdCellIndicesSorted ),
							thrust::device_ptr<uint>( pdCellIndicesSorted + numAgents ),
							thrust::device_ptr<uint>( pdAgentIndicesSorted ) );

	// Set all cells to empty.
	CUDA_SAFE_CALL( cudaMemset( m_pdCellStart, 0xffffffff, m_nCells * sizeof(uint) ) );

	KNNBinningReorderData<<< grid, block >>>(	pdPosition, pdDirection, pdSpeed,
												pdAgentIndicesSorted, pdCellIndicesSorted,
												pdPositionSorted, pdDirectionSorted, pdSpeedSorted,
												m_pdCellStart, m_pdCellEnd,
												numAgents
												);
	cutilCheckMsg( "KNNBinningReorderData failed" );
	CUDA_SAFE_CALL( cudaThreadSynchronize() );

	// Compute the size of shared memory needed for each block.
	size_t shMemSize = k * THREADSPERBLOCK * (sizeof(float) + sizeof(uint));

	KNNBinningKernel<<< grid, block, shMemSize >>>(	pdPositionSorted,
													pdAgentIndicesSorted, pdCellIndicesSorted,
													m_pdCellStart, m_pdCellEnd,
													pdKNNIndices, pdKNNDistances,
													k, 1, numAgents
											);
	cutilCheckMsg( "KNNBinningKernel failed." );
	CUDA_SAFE_CALL( cudaThreadSynchronize() );

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
}

void KNNBinningCUDA::close( void )
{
	// Unbind the texture.
	KNNBinningCUDAUnbindTexture();

	//CUDA_SAFE_CALL( cudaFree( m_pdCellIndices ) );
	//CUDA_SAFE_CALL( cudaFree( m_pdAgentIndices ) );

	CUDA_SAFE_CALL( cudaFree( m_pdCellStart ) );
	CUDA_SAFE_CALL( cudaFree( m_pdCellEnd ) );
}