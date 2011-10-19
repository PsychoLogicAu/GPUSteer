#include "CUDAKernelGlobals.cuh"

extern "C"
{

	__host__ void ComputeAverageCellVelocityBindTextures(	float4 const*	pdDirection,
															float const*	pdSpeed,
															uint const*		pdAgentIndicesSorted,
															uint const		numAgents
															);

	__host__ void ComputeAverageCellVelocityUnbindTextures( void );

	__global__ void ComputeCellDensity(						uint const*	pdCellStart,
															uint const*	pdCellEnd,
															uint *		pdAgentsPerCell,

															uint const	numCells
															);

	__global__ void ComputeAverageCellVelocity(				uint const*		pdCellStart,
															uint const*		pdCellEnd,

															float3 const*	pdAvgVelocity,

															uint const		numAgents,
															uint const		numCells
															);
}

// Compute the number of agents per cell.
__global__ void ComputeCellDensity(	uint const*	pdCellStart,
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

__host__ void ComputeAverageCellVelocityBindTextures(	float4 const*	pdDirection,
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

__host__ void ComputeAverageCellVelocityUnbindTextures( void )
{
	CUDA_SAFE_CALL( cudaUnbindTexture( texDirection ) );
	CUDA_SAFE_CALL( cudaUnbindTexture( texSpeed ) );
	CUDA_SAFE_CALL( cudaUnbindTexture( texAgentIndicesSorted ) );
}

// Compute the average velocity of agents in each cell.
__global__ void ComputeAverageCellVelocity(	uint const*		pdCellStart,
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
		uint i;
		for( i = shCellStart[ threadIdx.x ]; i < shCellEnd[ threadIdx.x ]; i++ )
		{
			uint const		agentIndex	= tex1Dfetch( texAgentIndicesSorted, i );

			float3 const	direction	= make_float3( tex1Dfetch( texDirection, agentIndex ) );
			float const		speed		= tex1Dfetch( texSpeed, agentIndex );

			float3 const	velocity	= float3_scalar_multiply( direction, speed );

			// Add into the running total.
			shAvgVelocity[ threadIdx.x ] = float3_add( shAvgVelocity[ threadIdx.x ], velocity );
		}
		
		// Divide by i to get the average.
		shAvgVelocity[ threadIdx.x ] = float3_scalar_divide( shAvgVelocity[ threadIdx.x ], i );
	}

	// Write the average velocities out to global memory.
	pdAvgVelocity[ index ] = shAvgVelocity[ threadIdx.x ];
}