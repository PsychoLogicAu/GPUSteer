#ifndef OPENSTEER_FLOCKINGCOMMON_CUH
#define OPENSTEER_FLOCKINGCOMMON_CUH

static __device__ bool	inBoidNeighborhood(	// Agent data.
											float3 const&	position,
											float3 const&	direction,

											float3 const&	otherPosition,

											float const&	minDistance,
											float const&	maxDistance,
											float const&	cosMaxAngle
										   )
{
	float3 const offset = float3_subtract( otherPosition, position );
	float const distance = float3_length( offset );

	// definitely in neighborhood if inside minDistance sphere
	if( distance < minDistance )
		return true;

	// definitely not in neighborhood if outside maxDistance sphere
	if( distance > maxDistance )
		return false;

	// otherwise, test angular offset from forward axis
	float3 const unitOffset = float3_scalar_divide( offset, distance );
	float const forwardness = float3_dot( direction, unitOffset );
	return forwardness > cosMaxAngle;
}

#endif
