#include "KNNBinningCUDA.cuh"

#include "../VectorUtils.cuh"

#include "CUDAKernelGlobals.cuh"

using namespace OpenSteer;

// Define the texture reference to access the appropriate bin_cell's index.
texture< uint, cudaTextureType3D, cudaReadModeElementType > texCellIndices;

// Fetch the bin from texBinCells at a given world {x,y,z} position.
#define CELLINDEX( pos ) ( tex3D( texCellIndices, pos.x, pos.z, pos.y ) )

// Kernel declarations.
extern "C"
{
	// Kernel to set initial bin indices of vehicles in the simulation.
	__global__ void KNNBinningBuildDB(	float3 const*	pdPosition,				// In:	Positions of each vehicle.
										size_t *		pdAgentIndices,			// Out:	Indices of each vehicle.
										size_t *		pdAgentCellIndices,		// Out:	Indices of the bin each vehicle is in.
										size_t const	numAgents				// In:	Number of agents in the simulation.
										);

	// Bind texCellIndices to the cudaArray.
	__host__ void KNNBinningCUDABindTexture( cudaArray * pCudaArray );
	// Unbind the texture.
	__host__ void KNNBinningCUDAUnbindTexture( void );

	__global__ void KNNBinningKernel(	float3 const*	pdPosition,			// In: Agent positions.
										size_t *		pdAgentIndices,		// In: (sorted) indices of each agent.
										size_t *		pdAgentCellIndices,	// In: (sorted) indices of the cell each agent is in.
										size_t const	k,					// In: Number of neighbors to consider.
										size_t const	radius,				// In: Maximum radius (in cells) to consider.
										size_t const	numAgents,			// In: Number of agents in the simulation.

										uint *			pdKNNIndices,		// Out: indices of K Nearest Neighbors in pdPosition.
										float *			pdKNNDistances,		// Out: distances of the K Nearest Neighbors in pdPosition.
										);
}

//__global__ void KNNBinningKernel(	float3 const*	pdPosition,			// In: Agent positions.
//									uint *			pdKNNIndices,		// Out: indices of K Nearest Neighbors in pdPosition.
//									float *			pdKNNDistances,		// Out: distances of the K Nearest Neighbors in pdPosition.
//									size_t const	k,					// In: Number of neighbors to consider.
//									size_t const	radius,				// In: Maximum radius (in cells) to consider.
//									size_t const	numAgents,			// In: Number of agents in the simulation.
//									)
{
	// Offset of this agent.
	int offset = (blockIdx.x * blockDim.x) + threadIdx.x;

	// Check bounds.
	if( offset > numAgents )
		return;


}

__host__ void KNNBinningCUDABindTexture( cudaArray * pdCudaArray )
{
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<uint>();

	texCellIndices.normalized = true;
	texCellIndices.filterMode = cudaFilterModePoint;
	texCellIndices.addressMode[0] = cudaAddressModeClamp;
	texCellIndices.addressMode[1] = cudaAddressModeClamp;
	texCellIndices.addressMode[2] = cudaAddressModeClamp;

	CUDA_SAFE_CALL( cudaBindTextureToArray( texCellIndices, pdCudaArray, channelDesc ) );
}

__host__ void KNNBinningCUDAUnbindTexture( void )
{
	CUDA_SAFE_CALL( cudaUnbindTexture( texCellIndices ) );
}

__global__ void KNNBinningBuildDB(	float3 const*	pdPosition,				// In:	Positions of each vehicle.
									size_t *		pdAgentIndices,			// Out:	Indices of each vehicle.
									size_t *		pdAgentBinIndices,		// Out:	Indices of the bin each vehicle is in.
									size_t const	numAgents				// In:	Number of agents in the simulation.
									)
{
	// Offset of this agent in the global array.
	int offset = (blockIdx.x * blockDim.x) + threadIdx.x;

	// Check bounds.
	if( offset >= numAgents )
		return;

	// Copy the positions to shared memory.
	__shared__ float3 shPosition[THREADSPERBLOCK];
	FLOAT3_COALESCED_READ( shPosition, pdPosition );
	//POSITION_SH( threadIdx.x ) = POSITION( offset );

	// Write the agent's cell index out to global memory.
	pdAgentBinIndices[offset] = CELLINDEX( POSITION_SH( threadIdx.x ) );

	// Write the agent's index out to global memory.
	pdAgentIndices[offset] = offset;
}