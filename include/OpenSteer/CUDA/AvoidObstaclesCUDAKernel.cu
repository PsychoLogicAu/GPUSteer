#include "CUDAKernelGlobals.cuh"

using namespace OpenSteer;

extern "C"
{
	__host__ void SteerToAvoidObstaclesKernelBindTextures(	float4 const*	pdObstaclePosition,
															float const*	pdObstacleRadius,
															uint const		numObstacles
															);
	__host__ void SteerToAvoidObstaclesKernelUnbindTextures( void );

	__global__ void SteerToAvoidObstaclesKernel(			uint const*		pdKNNIndices,			// In:	Indices of the K Nearest Obstacles.
															float const*	pdKNNDistances,			// In:	Distances to the K Nearest Obstacles.
															size_t const	k,
														
															float4 const*	pdPosition,				// In:	Agent positions.
															float4 const*	pdDirection,			// In:	Agent directions.
															float3 const*	pdSide,
															float3 const*	pdUp,
															float const*	pdRadius,				// In:	Agent radii.
															float const*	pdSpeed,				// In:	Agent speeds.

															float const		minTimeToCollision,
				
															float4 *		pdSteering,				// Out:	Agent steering vectors.
															
															uint const		numAgents,				// In:	Number of agents.
															uint const		numObstacles,			// In:	Number of obstacles.
															float const		fWeight,				// In:	Weight for this kernel

															uint *			pdAppliedKernels,
															uint const		doNotApplyWith
															);
}

texture< float4, cudaTextureType1D, cudaReadModeElementType >	texOPosition;
texture< float, cudaTextureType1D, cudaReadModeElementType >	texORadius;

typedef struct intersection
{
	bool intersects;
	float3 position;
	float distance;
} Intersection;

__inline__ __device__ void findNextIntersectionWithSphere(	float3 const* agentPosition, float3 const* side, float3 const* up, float3 const* direction, float const* agentRadius,
															float3 const* obstaclePosition, float const* obstacleRadius,
															Intersection *intersection)
{
	// This routine is based on the Paul Bourke's derivation in:
	//   Intersection of a Line and a Sphere (or circle)
	//   http://www.swin.edu.au/astronomy/pbourke/geometry/sphereline/

	float b, c, d, p, q, s;
	float3 lc;

	// initialize pathIntersection object
	intersection->intersects = false;
	intersection->position = *obstaclePosition;

	// find "local center" (lc) of sphere in boid's coordinate space
	float3 globalOffset = float3_subtract( *obstaclePosition, *agentPosition );
	lc = make_float3(	float3_dot( globalOffset, *side ),
						float3_dot( globalOffset, *up ),
						float3_dot( globalOffset, *direction ) 
						);

	// computer line-sphere intersection parameters
	b = -2 * lc.z;
	c = lc.x * lc.x + lc.y * lc.y + lc.z * lc.z - 
		(*obstacleRadius + *agentRadius) * (*obstacleRadius + *agentRadius);
	d = (b * b) - (4 * c);

	// when the path does not intersect the sphere
	if (d < 0)
		return;

	// otherwise, the path intersects the sphere in two points with
	// parametric coordinates of "p" and "q".
	// (If "d" is zero the two points are coincident, the path is tangent)
	s = sqrt(d);
	p = (-b + s) / 2;
	q = (-b - s) / 2;

	// both intersections are behind us, so no potential collisions
	if ((p < 0) && (q < 0))
		return; 

	// at least one intersection is in front of us
	intersection->intersects = true;
	intersection->distance =
		((p > 0) && (q > 0)) ?
		// both intersections are in front of us, find nearest one
		((p < q) ? p : q) :
		// otherwise only one intersections is in front, select it
		((p > 0) ? p : q);
	return;
}

__inline__ __device__ void LocalizeDirection( float3 const* globalDirection, float3 * localDirection, float3 const* side, float3 const* up, float3 const* direction )
{
	// Dot offset with local basis vectors to obtain local coordiantes.
	*localDirection =  make_float3(	float3_dot(*globalDirection, *side ),
									float3_dot(*globalDirection, *up ),
									float3_dot(*globalDirection, *direction )
									);
}

__inline__ __device__ void LocalizePosition( float3 const* globalPosition, float3 * localPosition, float3 const* position, float3 const* side, float3 const* up, float3 const* direction )
{
	// Global offset from local origin.
	float3 globalOffset = float3_subtract( *globalPosition, *position );

	LocalizeDirection( &globalOffset, localPosition, side, up, direction );
}

__host__ void SteerToAvoidObstaclesKernelBindTextures(	float4 const*	pdObstaclePosition,
														float const*	pdObstacleRadius,
														uint const		numObstacles
														)
{
	static cudaChannelFormatDesc const floatChannelDesc = cudaCreateChannelDesc< float >();
	static cudaChannelFormatDesc const float4ChannelDesc = cudaCreateChannelDesc< float4 >();

	CUDA_SAFE_CALL( cudaBindTexture( NULL, texOPosition, pdObstaclePosition, float4ChannelDesc, numObstacles * sizeof(float4) ) );
	CUDA_SAFE_CALL( cudaBindTexture( NULL, texORadius, pdObstacleRadius, floatChannelDesc, numObstacles * sizeof(float) ) );
}

__host__ void SteerToAvoidObstaclesKernelUnbindTextures( void )
{
	CUDA_SAFE_CALL( cudaUnbindTexture( texOPosition ) );
	CUDA_SAFE_CALL( cudaUnbindTexture( texORadius ) );
}

__global__ void SteerToAvoidObstaclesKernel(	uint const*		pdKNNIndices,			// In:	Indices of the K Nearest Obstacles.
												float const*	pdKNNDistances,			// In:	Distances to the K Nearest Obstacles.
												size_t const	k,
											
												float4 const*	pdPosition,				// In:	Agent positions.
												float4 const*	pdDirection,			// In:	Agent directions.
												float3 const*	pdSide,
												float3 const*	pdUp,
												float const*	pdRadius,				// In:	Agent radii.
												float const*	pdSpeed,				// In:	Agent speeds.

												float const		minTimeToCollision,
	
												float4 *		pdSteering,				// Out:	Agent steering vectors.
												
												uint const		numAgents,				// In:	Number of agents.
												uint const		numObstacles,			// In:	Number of obstacles.
												float const		fWeight,				// In:	Weight for this kernel

												uint *			pdAppliedKernels,
												uint const		doNotApplyWith
												)
{
	uint index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if( index >= numAgents )
		return;

	if( pdAppliedKernels[ index ] & doNotApplyWith )
		return;

	// Shared memory.
	extern __shared__ uint shKNNIndices[];
	float * shKNNDistances = (float*)shKNNIndices + (THREADSPERBLOCK*k);

	__shared__ float3 shPosition[THREADSPERBLOCK];
	__shared__ float3 shDirection[THREADSPERBLOCK];
	__shared__ float3 shSide[THREADSPERBLOCK];
	__shared__ float3 shUp[THREADSPERBLOCK];
	__shared__ float shRadius[THREADSPERBLOCK];
	__shared__ float shSpeed[THREADSPERBLOCK];
	__shared__ float3 shSteering[THREADSPERBLOCK];

	// Copy required from global memory.
	POSITION_SH( threadIdx.x )	= POSITION_F3( index );
	DIRECTION_SH( threadIdx.x ) = DIRECTION_F3( index );
	STEERING_SH( threadIdx.x )	= STEERING_F3( index );
	RADIUS_SH( threadIdx.x ) = RADIUS( index );
	SPEED_SH( threadIdx.x ) = SPEED( index );

	FLOAT3_GLOBAL_READ( shSide, pdSide );
	FLOAT3_GLOBAL_READ( shUp, pdUp );

	for( int i = 0; i < k; i++ )
	{
		//shKNNIndices[threadIdx.x*k + i] = pdKNNIndices[index*k + i];
		//shKNNDistances[threadIdx.x*k + i] = pdKNNDistances[index*k + i];
		shKNNIndices[ threadIdx.x + i * THREADSPERBLOCK ] = pdKNNIndices[ index + i * numAgents ];
		shKNNDistances[ threadIdx.x + i * THREADSPERBLOCK ] = pdKNNDistances[ index + i * numAgents ];
	}
	__syncthreads();

	float3 steering = { 0.f, 0.f, 0.f };
	
	float const			minDistanceToCollision = minTimeToCollision * SPEED_SH( threadIdx.x );
	Intersection		next, nearest;
	next.intersects		= false;
	nearest.intersects	= false;

	// For each obstacle in the KNN for this agent...
	for( uint i = 0; i < k; i++ )
	{
		//uint const obstacleIndex = shKNNIndices[threadIdx.x*k + i];
		uint const obstacleIndex = shKNNIndices[ threadIdx.x + i * THREADSPERBLOCK ];

		// Check for the final near obstacle.
		if( obstacleIndex >= numObstacles )
			break;

		float3 const	obstaclePosition	= make_float3( tex1Dfetch( texOPosition, obstacleIndex ) );
		float const		obstacleRadius		= tex1Dfetch( texORadius, obstacleIndex );

		findNextIntersectionWithSphere(	&POSITION_SH( threadIdx.x ), &SIDE_SH( threadIdx.x ), &UP_SH( threadIdx.x ), &DIRECTION_SH( threadIdx.x ), &RADIUS_SH( threadIdx.x ), 
										&obstaclePosition, &obstacleRadius,
										&next
										);

		if( (nearest.intersects == false) || ((next.intersects != false) && (next.distance < nearest.distance)) )
            nearest = next;
	}

	// when a nearest intersection was found
	if( (nearest.intersects != false) && (nearest.distance < minDistanceToCollision) )
	{
		// compute avoidance steering force: take offset from obstacle to me,
		// take the component of that which is lateral (perpendicular to my
		// forward direction), set length to maxForce, add a bit of forward
		// component (in capture the flag, we never want to slow down)
		float3 const obstacleOffset = float3_subtract( POSITION_SH( threadIdx.x ), nearest.position );

		steering = float3_perpendicularComponent( obstacleOffset, DIRECTION_SH( threadIdx.x ) );
	}

	__syncthreads();

	// Apply the weight.
	steering = float3_scalar_multiply( steering, fWeight );

	// Set the applied kernel bit.
	if( ! float3_equals( steering, float3_zero() ) )
		pdAppliedKernels[ index ] |= KERNEL_AVOID_OBSTACLES_BIT;

	// Add into the steering vector.
	STEERING_SH( threadIdx.x ) = float3_add( steering, STEERING_SH( threadIdx.x ) );

	// Write back to global memory.
	STEERING( index ) = STEERING_SH_F4( threadIdx.x );
}
