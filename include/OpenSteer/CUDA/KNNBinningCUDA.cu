#include "KNNBinningCUDA.cuh"

using namespace OpenSteer;

// Kernel file function prototypes.
extern "C"
{
	// Kernel to set initial bin indices of vehicles in the simulation.
	__global__ void KNNBinningBuildDB(	float3 const*	pdPosition,				// In:	Positions of each vehicle.
										size_t *		pdAgentIndices,			// Out:	Indices of each vehicle.
										size_t *		pdAgentBinIndices,		// Out:	Indices of the bin each vehicle is in.
										size_t const	numAgents				// In:	Number of agents in the simulation.
										);

	// Bind texCellIndices to the cudaArray.
	__host__ void KNNBinningCUDABindTexture( cudaArray * pCudaArray );
	__host__ void KNNBinningCUDAUnbindTexture( void );
}

KNNBinningCUDA::KNNBinningCUDA( VehicleGroup * pVehicleGroup, size_t const k )
:	AbstractCUDAKernel( pVehicleGroup ),
	m_k( k )
{

}

void KNNBinningCUDA::init( void )
{
	// Bind the cell indices texture.
	KNNBinningCUDABindTexture( m_pVehicleGroup->GetBinData().pdCellIndexArray() );

	// Allocate temporary device storage.
	CUDA_SAFE_CALL( cudaMalloc( &m_pdAgentCellIndices, getNumAgents() * sizeof(uint) ) );
	CUDA_SAFE_CALL( cudaMalloc( &m_pdAgentIndices, getNumAgents() * sizeof(uint) ) );
}

void KNNBinningCUDA::run( void )
{
	dim3 grid = gridDim();
	dim3 block = blockDim();

	// Gether the required device pointers.
	float3 const* pdPosition = m_pVehicleGroupData->pdPosition();

	// Call the kernel.
	KNNBinningBuildDB<<< grid, block >>>( pdPosition, m_pdAgentIndices, m_pdAgentCellIndices, getNumAgents() );

	CUDA_SAFE_CALL( cudaThreadSynchronize() );
}

void KNNBinningCUDA::close( void )
{
	// Unbind the texture.
	KNNBinningCUDAUnbindTexture();

	CUDA_SAFE_CALL( cudaFree( m_pdAgentCellIndices ) );
	CUDA_SAFE_CALL( cudaFree( m_pdAgentIndices ) );
}