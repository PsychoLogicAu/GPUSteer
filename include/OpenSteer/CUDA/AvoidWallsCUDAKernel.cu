#include "AvoidWallsCUDA.cuh"

#include "../AgentGroupData.cuh"
#include "../VectorUtils.cuh"

#include "CUDAKernelGlobals.cuh"

using namespace OpenSteer;

extern "C"
{

}

// Returns true if the points a,b,c are in counter-clockwise order.
__inline__ __device__ bool ccw( float3 const& a, float3 const& b, float3 const& c )
{
	return (c.z-a.z)*(b.x-a.x) > (b.z-a.z)*(c.x-a.x);
}

// Returns true if the line segments intersect, ignoring the degenerate cases which are specified here: http://compgeom.cs.uiuc.edu/~jeffe/teaching/373/notes/x06-sweepline.pdf
__inline__ __device__ bool lineSegmentIntersect( float3 const& a, float3 const& b, float3 const& c, float3 const& d )
{
	return (ccw(a,c,d) != ccw(b,c,d)) && (ccw(a,b,c) != ccw(a,b,d));
}

__global__ void AvoidWallsCUDAKernel(	// Agent data.
										float3 const*	pdPosition,
										float3 const*	pdDirection,
										float const*	pdSpeed,

										float3 *		pdSteering,

										// Wall data.
										float3 const*	pdLineStart,
										float3 const*	pdLineEnd,
										float3 const*	pdLineNormal,

										uint const*		pdKNLIndices,	// Indices of the K Nearest line segments...
										uint const		k,				// Number of lines in KNL.

										float const		lookAheadTime,
										float const		fWeight,

										size_t const	numAgents
										)
{
	int const index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if( index >= numAgents )
		return;

	extern __shared__ uint shKNLIndices[];

	__shared__ float3 shPosition[THREADSPERBLOCK];
	__shared__ float3 shDirection[THREADSPERBLOCK];
	__shared__ float shSpeed[THREADSPERBLOCK];

	// Copy this block's data to shared memory.
	FLOAT3_GLOBAL_READ( shPosition, pdPosition );
	FLOAT3_GLOBAL_READ( shDirection, pdDirection );
	SPEED_SH( threadIdx.x ) = SPEED( index );

	__syncthreads();

	// Get the agent's current velocity.
	float3 const vel = VELOCITY_SH( threadIdx.x );

	// Compute the position of the agent after lookAheadTime time.
	float3 const futurePosition = float3_add( POSITION_SH( threadIdx.x ), float3_scalar_multiply( VELOCITY_SH( threadIdx.x ), lookAheadTime ) );








}