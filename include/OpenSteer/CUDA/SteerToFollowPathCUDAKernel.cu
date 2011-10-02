#include "SteerToFollowPathCUDA.cuh"

#include "CUDAKernelGlobals.cuh"

#include "../AgentGroupData.cuh"

extern "C"
{
	__global__ void FollowPathCUDAKernel(	// Agent data.
											float3 const*	pdPosition,
											float3 const*	pdDirection,
											float const*	pdSpeed,

											float3 *		pdSteering,

											// Path data.
											float3 const*	pdPathPoints,
											float const*	pdPathLengths,
											float3 const*	pdPathNormals,
											uint const		numPoints,
											float const		radius,
											bool const		cyclic,
											float const		totalPathLength,

											float const		predictionTime,

											uint const		numAgents,
											float const		fWeight,
											uint *			pdAppliedKernels,
											uint const		doNotApplyWith
											);
}

__inline__ __device__ float pointToSegmentDistance(	float3 const&	point,
													float3 const&	ep0,
													float3 const&	ep1,

													float3 const&	segmentNormal,
													float const&	segmentLength,
													float &			segmentProjection,

													float3 &		chosen
													)
{
	// convert the test point to be "local" to ep0
	float3 const local = float3_subtract( point, ep0 );

	// find the projection of "local" onto "segmentNormal"
	segmentProjection = float3_dot( segmentNormal, local );

	// handle boundary cases: when projection is not on segment, the
	// nearest point is one of the endpoints of the segment
	if( segmentProjection < 0.f )
	{
		chosen = ep0;
		segmentProjection = 0;
		return float3_distance( point, ep0 );
	}
	if( segmentProjection > segmentLength )
	{
		chosen = ep1;
		segmentProjection = segmentLength;
		return float3_distance( point, ep1 );
	}

	// otherwise nearest point is projection point on segment
	chosen = float3_add( ep0, float3_scalar_multiply( segmentNormal, segmentProjection ) );
	return float3_distance( point, chosen );
}

__inline__ __device__ float mapPointToPathDistance(	float3 const&	point,
													uint const&		numPoints,

													float3 const*	pdPoints,
													float const*	pdLengths,
													float3 const*	pdNormals,

													float3 &		chosen
													)
{
	float d;
	float minDistance = FLT_MAX;
	float segmentLengthTotal = 0;
	float pathDistance = 0;

	for( uint i = 1; i < numPoints; i++ )
	{
		float const		segmentLength = pdLengths[i];
		float3 const	segmentNormal = pdNormals[i];
		float			segmentProjection;
		d = pointToSegmentDistance( point, pdPoints[i-1], pdPoints[i], segmentNormal, segmentLength, segmentProjection, chosen );
		if( d < minDistance )
		{
			minDistance = d;
			pathDistance = segmentLengthTotal + segmentProjection;
		}
		segmentLengthTotal += segmentLength;
	}

	// return distance along path of onPath point
	return pathDistance;
}

__inline__ __device__ float3 mapPointToPath(	float3 const&	point,
												float3 &		tangent,
												float &			outside,
												float const&	radius,

												float3 &		chosen,

												float3 const*	pdPoints,
												float const*	pdLengths,
												float3 const*	pdNormals,

												uint const&		numPoints
												)
{
	float d;
	float minDistance = FLT_MAX;
	float3 onPath;

	// loop over all segments, find the one nearest to the given point
	for( uint i = 1; i < numPoints; i++ )
	{
		float segmentLength = pdLengths[i];
		float3 segmentNormal = pdNormals[i];
		float segmentProjection;
		d = pointToSegmentDistance( point, pdPoints[i-1], pdPoints[i], segmentNormal, segmentLength, segmentProjection, chosen );
		if( d < minDistance )
		{
			minDistance = d;
			onPath = chosen;
			tangent = segmentNormal;
		}
	}

	// measure how far original point is outside the Pathway's "tube"
	outside = float3_distance( onPath, point ) - radius;

	// return point on path
	return onPath;
}

__inline__ __device__ float3 mapPathDistanceToPoint(	float const&	pathDistance,
														bool const&		cyclic,
														float const&	totalPathLength,

														float3 const*	pdPoints,
														float const*	pdLengths,
														uint const&		numPoints
														)
{
    // clip or wrap given path distance according to cyclic flag
    float remaining = pathDistance;
    if( cyclic )
    {
        remaining = (float)fmodf( pathDistance, totalPathLength );
    }
    else
    {
        if( pathDistance < 0.f )
			return pdPoints[0];
        if( pathDistance >= totalPathLength )
			return pdPoints[ numPoints - 1 ];
    }

    // step through segments, subtracting off segment lengths until
    // locating the segment that contains the original pathDistance.
    // Interpolate along that segment to find 3d point value to return.
    float3 result;
    for( uint i = 1; i < numPoints; i++ )
    {
        float segmentLength = pdLengths[i];
        if( segmentLength < remaining )
        {
            remaining -= segmentLength;
        }
        else
        {
            float ratio = remaining / segmentLength;
            result = interpolate( ratio, pdPoints[i-1], pdPoints[i] );
            break;
        }
    }

    return result;
}

__global__ void FollowPathCUDAKernel(	// Agent data.
										float3 const*	pdPosition,
										float3 const*	pdDirection,
										float const*	pdSpeed,

										float3 *		pdSteering,

										// Path data.
										float3 const*	pdPathPoints,
										float const*	pdPathLengths,
										float3 const*	pdPathNormals,
										uint const		numPoints,
										float const		radius,
										bool const		cyclic,
										float const		totalPathLength,

										float const		predictionTime,

										uint const		numAgents,
										float const		fWeight,
										uint *			pdAppliedKernels,
										uint const		doNotApplyWith
									 )
{
	int const index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if( index >= numAgents )
		return;

	if( pdAppliedKernels[ index ] & doNotApplyWith )
		return;

	float3 steering;

	// Shared memory.
	__shared__ float	shSpeed[THREADSPERBLOCK];
	__shared__ float3	shPosition[THREADSPERBLOCK];
	__shared__ float3	shDirection[THREADSPERBLOCK];
	__shared__ float3	shSteering[THREADSPERBLOCK];

	FLOAT3_GLOBAL_READ( shDirection, pdDirection );
	FLOAT3_GLOBAL_READ( shPosition, pdPosition );
	FLOAT3_GLOBAL_READ( shSteering, pdSteering );
	SPEED_SH( threadIdx.x ) = SPEED( index );
	__syncthreads();

	// our goal will be offset from our path distance by this amount
	const float pathDistanceOffset = predictionTime * SPEED_SH( threadIdx.x );

	// predict our future position
	float3 const futurePosition = float3_add( POSITION_SH( threadIdx.x ), float3_scalar_multiply( float3_scalar_multiply( DIRECTION_SH( threadIdx.x ), SPEED_SH( threadIdx.x ) ), predictionTime ) );

	float3 chosen;

	// measure the distance along the path of our current and predicted positions
	float const nowPathDistance = mapPointToPathDistance( POSITION_SH( threadIdx.x ), numPoints, pdPathPoints, pdPathLengths, pdPathNormals, chosen );
	const float futurePathDistance = mapPointToPathDistance( futurePosition, numPoints, pdPathPoints, pdPathLengths, pdPathNormals, chosen );

	// are we facing in the correction direction?
	const bool rightway = ((pathDistanceOffset > 0.f) ?
						   (nowPathDistance < futurePathDistance) :
						   (nowPathDistance > futurePathDistance));

	// find the point on the path nearest the predicted future position
	float3 tangent;
	float outside;
	float3 const onPath = mapPointToPath( futurePosition, tangent, outside, radius, chosen, pdPathPoints, pdPathLengths, pdPathNormals, numPoints );

	if( (outside < 0.f) && rightway )
	{
		steering = float3_zero();
	}
	else
    {
        // steer towards a target point obtained by adding pathDistanceOffset to our current path position
        float const targetPathDistance = nowPathDistance + pathDistanceOffset;
        float3 target = mapPathDistanceToPoint( targetPathDistance, cyclic, totalPathLength, pdPathPoints, pdPathLengths, numPoints );

        // Seek to the target on path.
        // Get the desired velocity.
		float3 const desiredVelocity = float3_subtract( target, POSITION_SH( threadIdx.x ) );

		// Set the steering vector.
		steering = float3_subtract( desiredVelocity, DIRECTION_SH( threadIdx.x ) );
    }

	// Normalize and apply the weight.
	steering = float3_scalar_multiply( float3_normalize( steering ), fWeight );

	// Set the applied kernel bit.
	if( ! float3_equals( steering, float3_zero() ) )
			pdAppliedKernels[ index ] |= KERNEL_FOLLOW_PATH_BIT;

	// Add into the steering vector.
	STEERING_SH( threadIdx.x ) = float3_add( steering, STEERING_SH( threadIdx.x ) );

	// Copy the steering vectors back to global memory.
	FLOAT3_GLOBAL_WRITE( pdSteering, shSteering );
}
