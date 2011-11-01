#include "CUDAKernelGlobals.cuh"

#include "../AgentGroup.h"
#include "KNNDatabase.cuh"

#include "CUDAGlobals.cuh"
#include "CUDAKernelGlobals.cuh"

using namespace OpenSteer;

extern "C"
{
	void ComputeCellDensity(		AgentGroup * pAgentGroup, uint * pAgentsPerCell );
	void ComputeAvgCellVelocity(	AgentGroup * pAgentGroup, float3 * pAvgCellVelocity );
}

void ComputeAvgCellVelocityBindTextures(	float4 const*	pdDirection,
											float const*	pdSpeed,
											uint const*		pdAgentIndicesSorted,
											uint const		numAgents
											);

void ComputeAvgCellVelocityUnbindTextures( void );

__global__ void ComputeCellDensityKernel(	uint const*	pdCellStart,
											uint const*	pdCellEnd,
											uint *		pdAgentsPerCell,

											uint const	numCells
											);

__global__ void ComputeAvgCellVelocityKernel(	uint const*		pdCellStart,
												uint const*		pdCellEnd,

												float3 *		pdAvgVelocity,

												uint const		numAgents,
												uint const		numCells
												);


dim3 dimGrid( uint const numElems )
{
return dim3( ( numElems + THREADSPERBLOCK - 1 ) / THREADSPERBLOCK );
}
dim3 dimBlock( void )
{
	return dim3( THREADSPERBLOCK );
}

// Computes the number of agents per cell and writes the reqult to pAgentsPerCell.
void ComputeCellDensity( AgentGroup * pAgentGroup, uint * phAgentsPerCell )
{
	// Ensure the device is synchronized.
	CUDA_SAFE_CALL( cudaDeviceSynchronize() );

	KNNDatabase &	rKNNDatabase = pAgentGroup->GetKNNDatabase();
	uint const		numCells = rKNNDatabase.cells();

	// One thread per grid cell.
	dim3 grid = dimGrid( numCells );
	dim3 block = dimBlock();

	// Temporary device storage.
	uint *		pdAgentsPerCell;
	cudaMalloc( &pdAgentsPerCell, numCells * sizeof(uint) );

	uint const*	pdCellStart = rKNNDatabase.pdCellStart();
	uint const*	pdCellEnd	= rKNNDatabase.pdCellEnd();

	ComputeCellDensityKernel<<< grid, block >>>(	pdCellStart,
													pdCellEnd,
													pdAgentsPerCell,
													numCells
													);
	CUDA_SAFE_CALL( cudaDeviceSynchronize() );

	cudaMemcpy(	phAgentsPerCell, pdAgentsPerCell, numCells * sizeof(uint), cudaMemcpyDeviceToHost );
	cudaFree( pdAgentsPerCell );
}

// Computes the average velocity of agents in each cell and writes the reqults to pAvgCellVelocity.
void ComputeAvgCellVelocity( AgentGroup * pAgentGroup, float3 * phAvgVelocity )
{
	// Ensure the device is synchronized.
	CUDA_SAFE_CALL( cudaDeviceSynchronize() );

	KNNDatabase &	rKNNDatabase	= pAgentGroup->GetKNNDatabase();
	uint const		numCells		= rKNNDatabase.cells();
	uint const		numAgents		= pAgentGroup->Size();

	// One thread per grid cell.
	dim3 grid	= dimGrid( numCells );
	dim3 block	= dimBlock();

	// Temporary device storage.
	float3 *		pdAvgVelocity;
	cudaMalloc( &pdAvgVelocity, numCells * sizeof(float3) );

	float4 const*	pdDirection				= pAgentGroup->GetAgentGroupData().pdDirection();
	float const*	pdSpeed					= pAgentGroup->GetAgentGroupData().pdSpeed();
	uint const*		pdAgentIndicesSorted	= rKNNDatabase.pdAgentIndicesSorted();

	uint const*		pdCellStart				= rKNNDatabase.pdCellStart();
	uint const*		pdCellEnd				= rKNNDatabase.pdCellEnd();

	// Bind the textures.
	ComputeAvgCellVelocityBindTextures(	pdDirection, 
										pdSpeed,
										pdAgentIndicesSorted,
										numAgents
										);

	// Call the kernel.
	ComputeAvgCellVelocityKernel<<< grid, block >>>(	pdCellStart,
														pdCellEnd,
														pdAvgVelocity,
														numAgents,
														numCells
														);
	CUDA_SAFE_CALL( cudaDeviceSynchronize() );

	// Unbind the textures
	ComputeAvgCellVelocityUnbindTextures();

	// Copy the device data to the host.
	cudaMemcpy( phAvgVelocity, pdAvgVelocity, numCells * sizeof(float3), cudaMemcpyDeviceToHost );
	cudaFree( pdAvgVelocity );
}

// Compute the number of agents per cell.
__global__ void ComputeCellDensityKernel(	uint const*	pdCellStart,
											uint const*	pdCellEnd,
											uint *		pdAgentsPerCell,

											uint const	numCells
											)
{
	// One thread per cell.
	int const index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if( index >= numCells )
		return;

	__shared__ uint	shCellStart[ THREADSPERBLOCK ];
	__shared__ uint	shCellEnd[ THREADSPERBLOCK ];

	__shared__ uint shAgentsPerCell[ THREADSPERBLOCK ];

	shCellStart[ threadIdx.x ]	= pdCellStart[ index ];
	shCellEnd[ threadIdx.x ]	= pdCellEnd[ index ];

	if( UINT_MAX == shCellStart[ threadIdx.x ] )
	{
		// This cell is empty.
		shAgentsPerCell[ threadIdx.x ] = 0;
	}
	else
	{
		shAgentsPerCell[ threadIdx.x ] = shCellEnd[ threadIdx.x ] - shCellStart[ threadIdx.x ];
	}

	// Write results to global memory.
	pdAgentsPerCell[ index ] = shAgentsPerCell[ threadIdx.x ];
}

texture< float4, cudaTextureType1D, cudaReadModeElementType >	texDirection;
texture< float, cudaTextureType1D, cudaReadModeElementType >	texSpeed;
texture< uint, cudaTextureType1D, cudaReadModeElementType >		texAgentIndicesSorted;

__host__ void ComputeAvgCellVelocityBindTextures(	float4 const*	pdDirection,
											float const*	pdSpeed,
											uint const*		pdAgentIndicesSorted,
											uint const		numAgents
											)
{
	static cudaChannelFormatDesc const float4ChannelDesc = cudaCreateChannelDesc< float4 >(); 
	static cudaChannelFormatDesc const floatChannelDesc = cudaCreateChannelDesc< float >(); 
	static cudaChannelFormatDesc const uintChannelDesc = cudaCreateChannelDesc< uint >(); 

	CUDA_SAFE_CALL( cudaBindTexture( 0, texDirection, pdDirection, float4ChannelDesc, numAgents * sizeof(float4) ) );
	CUDA_SAFE_CALL( cudaBindTexture( 0, texSpeed, pdSpeed, floatChannelDesc, numAgents * sizeof(float) ) );
	CUDA_SAFE_CALL( cudaBindTexture( 0, texAgentIndicesSorted, pdAgentIndicesSorted, uintChannelDesc, numAgents * sizeof(uint) ) );
}

__host__ void ComputeAvgCellVelocityUnbindTextures( void )
{
	CUDA_SAFE_CALL( cudaUnbindTexture( texDirection ) );
	CUDA_SAFE_CALL( cudaUnbindTexture( texSpeed ) );
	CUDA_SAFE_CALL( cudaUnbindTexture( texAgentIndicesSorted ) );
}

// Compute the average velocity of agents in each cell.
__global__ void ComputeAvgCellVelocityKernel(	uint const*		pdCellStart,
												uint const*		pdCellEnd,

												float3 *		pdAvgVelocity,

												uint const		numAgents,
												uint const		numCells
												)
{
	int const index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if( index >= numCells )
		return;

	__shared__ uint		shCellStart[ THREADSPERBLOCK ];
	__shared__ uint		shCellEnd[ THREADSPERBLOCK ];
	__shared__ float3	shAvgVelocity[ THREADSPERBLOCK ];

	shCellStart[ threadIdx.x ]		= pdCellStart[ index ];
	shCellEnd[ threadIdx.x ]		= pdCellEnd[ index ];
	shAvgVelocity[ threadIdx.x ]	= float3_zero();
	
	if( UINT_MAX != shCellStart[ threadIdx.x ] )
	{
		for( uint i = shCellStart[ threadIdx.x ]; i < shCellEnd[ threadIdx.x ]; i++ )
		{
			uint const		agentIndex	= tex1Dfetch( texAgentIndicesSorted, i );

			float3 const	direction	= make_float3( tex1Dfetch( texDirection, agentIndex ) );
			float const		speed		= tex1Dfetch( texSpeed, agentIndex );

			float3 const	velocity	= float3_scalar_multiply( direction, speed );

			// Add into the running total.
			shAvgVelocity[ threadIdx.x ] = float3_add( shAvgVelocity[ threadIdx.x ], velocity );
		}
		
		// Divide by i to get the average.
		shAvgVelocity[ threadIdx.x ] = float3_scalar_divide( shAvgVelocity[ threadIdx.x ], (shCellEnd[ threadIdx.x ] - shCellStart[ threadIdx.x ]) );
	}

	// Write the average velocities out to global memory.
	pdAvgVelocity[ index ] = shAvgVelocity[ threadIdx.x ];
}
