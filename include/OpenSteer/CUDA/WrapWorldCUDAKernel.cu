#include "CUDAKernelGlobals.cuh"

extern "C"
{
	__global__ void WrapWorldKernel(	float4 *		pdPosition,
										float3 const	worldSize,
										uint const		numAgents
										);
}

__global__ void WrapWorldKernel(	float4 *		pdPosition,
									float3 const	worldSize,
									uint const		numAgents
									)
{
	int const index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if( index >= numAgents )
		return;

	float3 const halfWorldDim = float3_scalar_multiply( worldSize, 0.5f );

	__shared__ float3	shPosition[ THREADSPERBLOCK ];

	POSITION_SH( threadIdx.x ) = POSITION_F3( index );

	if( POSITION_SH( threadIdx.x ).x < -halfWorldDim.x )
		POSITION_SH( threadIdx.x ).x = halfWorldDim.x;
	if( POSITION_SH( threadIdx.x ).x > halfWorldDim.x )
		POSITION_SH( threadIdx.x ).x = -halfWorldDim.x;

	if( POSITION_SH( threadIdx.x ).y < -halfWorldDim.y )
		POSITION_SH( threadIdx.x ).y = halfWorldDim.y;
	if( POSITION_SH( threadIdx.x ).y > halfWorldDim.y )
		POSITION_SH( threadIdx.x ).y = -halfWorldDim.y;

	//if( POSITION_SH( threadIdx.x ).z < -halfWorldDim.z )
	//	POSITION_SH( threadIdx.x ).z = halfWorldDim.z;
	//if( POSITION_SH( threadIdx.x ).z > halfWorldDim.z )
	//	POSITION_SH( threadIdx.x ).z = -halfWorldDim.z;

	if( POSITION_SH( threadIdx.x ).z < 0 )
		POSITION_SH( threadIdx.x ).z = halfWorldDim.z + 100.f;
	//if( POSITION_SH( threadIdx.x ).z > halfWorldDim.z )
	//	POSITION_SH( threadIdx.x ).z = 0;

	POSITION( index ) = POSITION_SH_F4( threadIdx.x );
}
