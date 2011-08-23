#include "KNNBinningCUDA.cuh"

using namespace OpenSteer;

#include <thrust/sort.h>

// Kernel file function prototypes.
extern "C"
{
	// Bind texCellIndices to the cudaArray.
	__host__ void KNNBinningCUDABindTexture( cudaArray * pCudaArray );
	__host__ void KNNBinningCUDAUnbindTexture( void );

	// Kernel to set initial bin indices of vehicles in the simulation.
	__global__ void KNNBinningBuildDB(	float3 const*	pdPosition,				// In:	Positions of each vehicle.
										size_t *		pdAgentIndices,			// Out:	Indices of each vehicle.
										size_t *		pdAgentBinIndices,		// Out:	Indices of the bin each vehicle is in.
										size_t const	numAgents				// In:	Number of agents in the simulation.
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
}

KNNBinningCUDA::KNNBinningCUDA( VehicleGroup * pVehicleGroup, size_t const k )
:	AbstractCUDAKernel( pVehicleGroup ),
	m_k( k )
{
	m_nCells = m_pVehicleGroup->GetBinData().getNumCells();
}

void KNNBinningCUDA::init( void )
{
	// Bind the cell indices texture.
	KNNBinningCUDABindTexture( m_pVehicleGroup->GetBinData().pdCellIndexArray() );

	// Allocate temporary device storage.
	CUDA_SAFE_CALL( cudaMalloc( &m_pdCellIndices, getNumAgents() * sizeof(uint) ) );
	CUDA_SAFE_CALL( cudaMalloc( &m_pdAgentIndices, getNumAgents() * sizeof(uint) ) );

	CUDA_SAFE_CALL( cudaMalloc( &m_pdCellStart, m_nCells * sizeof(uint) ) );
	CUDA_SAFE_CALL( cudaMalloc( &m_pdCellEnd, m_nCells * sizeof(uint) ) );
}

void KNNBinningCUDA::run( void )
{
	dim3 grid = gridDim();
	dim3 block = blockDim();

	size_t const	numAgents = getNumAgents();

	// Gather the required device pointers.
	float3 const*	pdPosition = m_pVehicleGroupData->pdPosition();
	float3 const*	pdDirection = m_pVehicleGroupData->pdForward();
	float const*	pdSpeed = m_pVehicleGroupData->pdSpeed();

	// Pointers to output data.
	uint *			pdKNNIndices = m_pVehicleGroup->GetNearestNeighborData().pdKNNIndices();
	float *			pdKNNDistances = m_pVehicleGroup->GetNearestNeighborData().pdKNNDistances();

	float3 *		pdPositionSorted = m_pVehicleGroup->GetNearestNeighborData().pdKNNPositions();
	float3 *		pdDirectionSorted = m_pVehicleGroup->GetNearestNeighborData().pdKNNDirections();
	float *			pdSpeedSorted = m_pVehicleGroup->GetNearestNeighborData().pdKNNSpeeds();

	// Call the kernel.
	KNNBinningBuildDB<<< grid, block >>>( pdPosition, m_pdAgentIndices, m_pdCellIndices, numAgents );
	cutilCheckMsg( "KNNBinningBuildDB failed." );
	// Wait for the kernel to complete.
	CUDA_SAFE_CALL( cudaThreadSynchronize() );

	// Sort m_pAgentIndices on m_pdCellIndices using thrust.
	thrust::sort_by_key(	thrust::device_ptr<uint>( m_pdCellIndices ),
							thrust::device_ptr<uint>( m_pdCellIndices + numAgents ),
							thrust::device_ptr<uint>( m_pdAgentIndices ) );

	// Set all cells to empty.
	CUDA_SAFE_CALL( cudaMemset( m_pdCellStart, 0xffffffff, m_nCells * sizeof(uint) ) );

	KNNBinningReorderData<<< grid, block >>>(	pdPosition, pdDirection, pdSpeed,
												m_pdAgentIndices, m_pdCellIndices,
												pdPositionSorted, pdDirectionSorted, pdSpeedSorted,
												m_pdCellStart, m_pdCellEnd,
												numAgents
												);
	cutilCheckMsg( "KNNBinningReorderData failed" );
	// Wait for the kernel to complete.
	CUDA_SAFE_CALL( cudaThreadSynchronize() );




	//KNNBinningKernel<<< grid, block >>>(  )
	//cutilCheckMsg( "KNNBinningBuildDB failed." );

	//// Wait for the kernel to complete.
	//CUDA_SAFE_CALL( cudaThreadSynchronize() );
}

void KNNBinningCUDA::close( void )
{
	// Unbind the texture.
	KNNBinningCUDAUnbindTexture();

	CUDA_SAFE_CALL( cudaFree( m_pdCellIndices ) );
	CUDA_SAFE_CALL( cudaFree( m_pdAgentIndices ) );

	CUDA_SAFE_CALL( cudaFree( m_pdCellStart ) );
	CUDA_SAFE_CALL( cudaFree( m_pdCellEnd ) );
}