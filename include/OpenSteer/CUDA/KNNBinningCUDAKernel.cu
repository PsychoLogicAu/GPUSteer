#include "KNNBinningCUDA.cuh"

#include "../VectorUtils.cuh"

#include "CUDAKernelGlobals.h"

using namespace OpenSteer;

texture< knn_bin_data > texBins;



// Kernel declarations.
extern "C"
{
	// TODO: this might be better being done offline, once per map.
	__global__ void KNNBinningCreateBins(	knn_bin_data * pdBinData,
											float const fWorldSizeX,
											float const fWorldSizeY,
											size_t const numBinsX,
											size_t const numBinsY
											);

	__global__ void KNNBinningBuildDB(	float3 const*	pdPosition,
										size_t *		pdVehicleIndices,
										size_t const*	pdVehicleBinIDs,
										size_t const	numAgents
										);

}

__global__ void KNNBinningCreateBins(	knn_bin_data * pdBinData,
										float const fWorldSizeX,
										float const fWorldSizeY,
										size_t const numBinsX,
										size_t const numBinsY
										)
{
	// X and Y coords of this bin.
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	// Compute the offset of this bin in the array.
	int offset = x + y * blockDim.x * gridDim.x;

	// Declare shared memory to hold the 
}