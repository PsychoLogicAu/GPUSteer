#include "SteerForAlignmentCUDA.cuh"

#include "CUDAKernelGlobals.cuh"

#include "FlockingCommon.cuh"

extern "C"
{
	__global__ void SteerForAlignmentCUDAKernel(	float3 const*	pdAPosition,
													float3 const*	pdADirection,
													float3 *		pdASteering,
													size_t const	numA,

													uint const*		pdKNNIndices,
													size_t const	k,

													float3 const*	pdBPosition,
													float3 const*	pdBDirection,
													uint const		numB,

													float const		maxDistance,
													float const		cosMaxAngle,

													float const		fWeight,
													uint *			pdAppliedKernels,
													uint const		doNotApplyWith
													);
}

__global__ void SteerForAlignmentCUDAKernel(	float3 const*	pdAPosition,
												float3 const*	pdADirection,
												float3 *		pdASteering,
												size_t const	numA,

												uint const*		pdKNNIndices,
												size_t const	k,

												float3 const*	pdBPosition,
												float3 const*	pdBDirection,
												uint const		numB,

												float const		maxDistance,
												float const		cosMaxAngle,

												float const		fWeight,
												uint *			pdAppliedKernels,
												uint const		doNotApplyWith
												)
{
	int const index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if( index >= numA )
		return;

	if( pdAppliedKernels[ index ] & doNotApplyWith )
		return;

	extern __shared__ uint shKNNIndices[];

	__shared__ float3 shSteering[THREADSPERBLOCK];
	__shared__ float3 shPosition[THREADSPERBLOCK];
	__shared__ float3 shDirection[THREADSPERBLOCK];

	// Copy required from global memory.
	FLOAT3_GLOBAL_READ( shSteering, pdASteering );
	FLOAT3_GLOBAL_READ( shPosition, pdAPosition );
	FLOAT3_GLOBAL_READ( shDirection, pdADirection );

	for( int i = 0; i < k; i++ )
		shKNNIndices[threadIdx.x*k + i] = pdKNNIndices[index*k + i];
	__syncthreads();

	    // steering accumulator and count of neighbors, both initially zero
	float3 steering = { 0.f, 0.f, 0.f };
    uint neighbors = 0;

    // For each agent in this agent's KNN neighborhood...
	for( uint i = 0; i < k; i++ )
	{
		uint BIndex = shKNNIndices[threadIdx.x * k + i];

		// Check for end of KNN.
		if( BIndex >= numB )
			break;

		// TODO: texture memory.
		float3 const bPosition = pdBPosition[ BIndex ];
		float3 const bDirection = pdBDirection[ BIndex ];

		if( inBoidNeighborhood( POSITION_SH( threadIdx.x ), DIRECTION_SH( threadIdx.x ), bPosition, maxDistance, cosMaxAngle ) )
		{
			// accumulate sum of neighbor's positions
			steering = float3_add( steering, bDirection );

			// count neighbors
			neighbors++;
		}
	}

	if( neighbors > 0 )
		steering = float3_normalize( float3_subtract( float3_scalar_divide( steering, (float)neighbors ), DIRECTION_SH( threadIdx.x ) ) );

	// Apply the weight.
	steering = float3_scalar_multiply( steering, fWeight );

	// Set the applied kernel bit.
	if( ! float3_equals( steering, float3_zero() ) )
		pdAppliedKernels[ index ] |= KERNEL_ALIGNMENT_BIT;

	// Add into the steering vector.
	STEERING_SH( threadIdx.x ) = float3_add( steering, STEERING_SH( threadIdx.x ) );

	// Write back to global memory.
	FLOAT3_GLOBAL_WRITE( pdASteering, shSteering );
}
