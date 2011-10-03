#ifndef OPENSTEER_FLOCKINGCOMMON_CUH
#define OPENSTEER_FLOCKINGCOMMON_CUH

__inline__ __device__ bool	inBoidNeighborhood(	// Agent data.
												float3 const&	position,
												float3 const&	direction,

												float3 const&	otherPosition,

												float const&	maxDistance,
												float const&	cosMaxAngle
											   )
{
	return true;

	float3 const offset = float3_subtract( otherPosition, position );
	float const distanceSquared = float3_lengthSquared( offset );

	// definitely not in neighborhood if outside maxDistance sphere
	if( distanceSquared > (maxDistance * maxDistance) )
		return false;

	// otherwise, test angular offset from forward axis
	float3 const unitOffset = float3_scalar_divide( offset, sqrtf( distanceSquared ) );
	float const forwardness = float3_dot( direction, unitOffset );
	return forwardness > cosMaxAngle;
}

#endif
