#ifndef OPENSTEER_VECTORUTILS_CU
#define OPENSTEER_VECTORUTILS_CU

// CUDA includes
#include <vector_types.h>
#include <vector_functions.h>

#include "Utilities.h"

static __inline__ __host__ __device__ float3 make_float3( float4 const& f4 )
{
	return make_float3( f4.x, f4.y, f4.z );
}

static __inline__ __host__ __device__ float4 make_float4( float3 const& f3, float const& w )
{
	return make_float4( f3.x, f3.y, f3.z, w );
}

static inline __host__ std::ostream& operator<<(std::ostream &os, const float3 &v)
{
	os << "[" << v.x << ", " << v.y << ", " << v.z << "]";
	return os;
}

// vector addition
static __inline__ __host__ __device__ float3 float3_add( float3 const& lhs, float3 const& rhs )
{
	return make_float3(lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z);
}
static __inline__ __host__ __device__ float4 float4_add( float4 const& lhs, float4 const& rhs )
{
	return make_float4( lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z, lhs.w + rhs.w );
}

// vector subtraction
static __inline__ __host__ __device__ float3 float3_subtract(float3 const& lhs, float3 const& rhs)
{
	return make_float3(lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z);
}
static __inline__ __host__ __device__ float4 float4_subtract( float4 const& lhs, float4 const& rhs )
{
	return make_float4( lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z, lhs.w - rhs.w );
}

// unary minus
static __inline__ __host__ __device__  float3 float3_minus(float3 const& v)
{
	return make_float3(-v.x, -v.y, -v.z);
}
static __inline__ __host__ __device__  float4 float4_minus( float4 const& v )
{
	return make_float4( -v.x, -v.y, -v.z, -v.w );
}

// vector times scalar product (scale length of vector times argument)
static __inline__ __host__ __device__  float3 float3_scalar_multiply(float3 const& v, const float s)
{
	return make_float3(v.x * s, v.y * s, v.z * s);
}
static __inline__ __host__ __device__  float4 float4_scalar_multiply( float4 const& v, const float s )
{
	return make_float4( v.x * s, v.y * s, v.z * s, v.w * s );
}

// vector divided by a scalar (divide length of vector by argument)
static __inline__ __host__ __device__  float3 float3_scalar_divide(float3 const& v, const float s)
{
	return make_float3(v.x / s, v.y / s, v.z / s);
}
static __inline__ __host__ __device__  float4 float4_scalar_divide( float4 const& v, const float s)
{
	return make_float4( v.x / s, v.y / s, v.z / s, v.w / s );
}

// dot product
static __inline__ __host__ __device__  float float3_dot(float3 const& lhs, float3 const& rhs)
{
	return (lhs.x * rhs.x) + (lhs.y * rhs.y) + (lhs.z * rhs.z);
}
static __inline__ __host__ __device__  float float4_dot( float4 const& lhs, float4 const& rhs )
{
	return (lhs.x * rhs.x) + (lhs.y * rhs.y) + (lhs.z * rhs.z) + (lhs.w * rhs.w);
}

// length squared
static __inline__ __host__ __device__  float float3_lengthSquared(float3 const& v)
{
	return float3_dot(v, v);
}
static __inline__ __host__ __device__  float float4_lengthSquared( float4 const& v )
{
	return float4_dot( v, v );
}

// length
static __inline__ __host__ __device__  float float3_length(float3 const& v)
{
	return sqrt(float3_lengthSquared(v));
}
static __inline__ __host__ __device__  float float4_length( float4 const& v )
{
	return sqrt( float4_lengthSquared( v ) );
}

// normalize
static __inline__ __host__ __device__  float3 float3_normalize (float3 const& v)
{
    // skip divide if length is zero
    const float len = float3_length(v);
    return (len > 0) ? float3_scalar_divide(v, len) : v;
}
static __inline__ __host__ __device__  float4 float4_normalize( float4 const& v )
{
    // skip divide if length is zero
    const float len = float4_length( v );
    return (len > 0) ? float4_scalar_divide( v, len ) : v;
}

// cross product
static __inline__ __host__ __device__  float3 float3_cross(float3 const& lhs, float3 const& rhs)
{
    return make_float3((lhs.y * rhs.z) - (lhs.z * rhs.y),
                  (lhs.z * rhs.x) - (lhs.x * rhs.z),
                  (lhs.x * rhs.y) - (lhs.y * rhs.x));
}

// equality/inequality
static __inline__ __host__ __device__  bool float3_equals(float3 const& lhs, float3 const& rhs)
{
	return lhs.x == rhs.x && lhs.y == rhs.y && lhs.z == rhs.z;
}
static __inline__ __host__ __device__  bool float4_equals( float4 const& lhs, float4 const& rhs)
{
	return lhs.x == rhs.x && lhs.y == rhs.y && lhs.z == rhs.z && lhs.w == rhs.w;
}

// Compute the euclidean distance between two points.
static __inline__ __host__ __device__  float float3_distance(float3 const& lhs, float3 const& rhs)
{
	return float3_length(float3_subtract(lhs, rhs));
}
static __inline__ __host__ __device__  float float4_distance( float4 const& lhs, float4 const& rhs)
{
	return float4_length( float4_subtract( lhs, rhs ) );
}

// return component of vector parallel to a unit basis vector
// (IMPORTANT NOTE: assumes "unitBasis" has unit magnitude (length==1))
static __inline__ __host__ __device__  float3 float3_parallelComponent(float3 const& v, float3 const& unitBasis)
{
	const float projection = float3_dot(v, unitBasis);

	return float3_scalar_multiply(unitBasis, projection);
}

// return component of vector perpendicular to a unit basis vector
// (IMPORTANT NOTE: assumes "basis" has unit magnitude (length==1))
static __inline__ __host__ __device__  float3 float3_perpendicularComponent(float3 const& v, float3 const& unitBasis)
{
	return float3_subtract(v, float3_parallelComponent(v, unitBasis));
}

// clamps the length of a given vector to maxLength.  If the vector is
// shorter its value is returned unaltered, if the vector is longer
// the value returned has length of maxLength and is paralle to the
// original input.
static __inline__ __host__ __device__  float3 float3_truncateLength(float3 const& v, const float maxLength)
{
    const float maxLengthSquared = maxLength * maxLength;
    const float vecLengthSquared = float3_lengthSquared(v);
    if (vecLengthSquared <= maxLengthSquared)
        return v;
	else
		return float3_scalar_multiply(v, maxLength / sqrt(vecLengthSquared));
}

static __inline__ __host__ __device__  float3 float3_setYtoZero(float3 const& v)
{
	return make_float3(v.x, 0, v.z);
}

static __inline__ __host__ __device__  float3 float3_zero()
{
	return make_float3(0, 0, 0);
}

static __inline__ __host__ __device__  float3 float3_up()
{
	return make_float3(0, 1, 0);
}

static __inline__ __host__ __device__  float3 float3_forward()
{
	return make_float3(0, 0, 1);
}

static __inline__ __host__ __device__ float3 float3_findPerpendicularIn3d(float3 const& direction)
{
    // to be filled in:
    float3 quasiPerp;  // a direction which is "almost perpendicular"
    float3 result;     // the computed perpendicular to be returned

    // three mutually perpendicular basis vectors
    const float3 i = make_float3(1, 0, 0);
    const float3 j = make_float3(0, 1, 0);
    const float3 k = make_float3(0, 0, 1);

    // measure the projection of "direction" onto each of the axes
    //const float id = i.dot (direction);
	const float id = float3_dot(i, direction);//  i.dot (direction);
    //const float jd = j.dot (direction);
	const float jd = float3_dot(j, direction);
    //const float kd = k.dot (direction);
	const float kd = float3_dot(k, direction);

    // set quasiPerp to the basis which is least parallel to "direction"
    if ((id <= jd) && (id <= kd))
    {
        quasiPerp = i;               // projection onto i was the smallest
    }
    else
    {
        if ((jd <= id) && (jd <= kd))
            quasiPerp = j;           // projection onto j was the smallest
        else
            quasiPerp = k;           // projection onto k was the smallest
    }

    // return the cross product (direction x quasiPerp)
    // which is guaranteed to be perpendicular to both of them
	result = float3_cross(direction, quasiPerp);
    return result;
}

// rotate this vector about the global Y (up) axis by the given angle
static __inline__ __host__ __device__ float3 float3_rotateAboutGlobalY (float3 const& v, float angle, float& _sin, float& _cos) 
{
    // is both are zero, they have not be initialized yet
    if (_sin == 0 && _cos == 0)
    {
        _sin = sinf(angle);
        _cos = cosf(angle);
    }
	return make_float3((v.x * _cos) + (v.z * _sin),
					   (v.y),
					   (v.z * _cos) - (v.x * _sin));
}

static __inline__ __host__ __device__ float float3_distanceFromLine (float3 const& point,
                                   float3 const& lineOrigin,
                                   float3 const& lineUnitTangent)
{
    const float3 offset = float3_subtract(point, lineOrigin);
	const float3 perp = float3_perpendicularComponent(offset, lineUnitTangent);
    return float3_length(perp);
}

static __inline__ __host__ float3 float3_RandomVectorInUnitRadiusSphere (void)
{
    float3 v;

    do
    {
		v.x = OpenSteer::frandom01()*2 - 1;
        v.y = OpenSteer::frandom01()*2 - 1;
		v.z = OpenSteer::frandom01()*2 - 1;
    }
    while (float3_length(v) >= 1);

    return v;
}

static __inline__ __host__ float3 float3_RandomUnitVectorOnXZPlane (void)
{
	return float3_normalize(float3_setYtoZero(float3_RandomVectorInUnitRadiusSphere()));
}

static __inline__ __host__ __device__ float3 interpolate (float alpha, float3 const& x0, float3 const& x1)
{
	return float3_add(x0, float3_scalar_multiply(float3_subtract(x1, x0), alpha));
}

// ----------------------------------------------------------------------------
// Does a "ceiling" or "floor" operation on the angle by which a given vector
// deviates from a given reference basis vector.  Consider a cone with "basis"
// as its axis and slope of "cosineOfConeAngle".  The first argument controls
// whether the "source" vector is forced to remain inside or outside of this
// cone.  Called by vecLimitMaxDeviationAngle and vecLimitMinDeviationAngle.


static __inline__ __host__ __device__ float3 vecLimitDeviationAngleUtility (const bool insideOrOutside,
                                          float3 const& source,
                                          const float cosineOfConeAngle,
                                          float3 const& basis)
{
    // immediately return zero length input vectors
    float sourceLength = float3_length(source);
    if (sourceLength == 0)
		return source;

    // measure the angular diviation of "source" from "basis"
    const float3 direction = float3_scalar_divide(source, sourceLength);
    float cosineOfSourceAngle = float3_dot(direction, basis);

    // Simply return "source" if it already meets the angle criteria.
    // (note: we hope this top "if" gets compiled out since the flag
    // is a constant when the function is inlined into its caller)
    if (insideOrOutside)
    {
		// source vector is already inside the cone, just return it
		if (cosineOfSourceAngle >= cosineOfConeAngle)
			return source;
    }
    else
    {
		// source vector is already outside the cone, just return it
		if (cosineOfSourceAngle <= cosineOfConeAngle)
			return source;
    }

    // find the portion of "source" that is perpendicular to "basis"
    const float3 perp = float3_perpendicularComponent(source, basis);

    // normalize that perpendicular
    const float3 unitPerp = float3_normalize(perp);

    // construct a new vector whose length equals the source vector,
    // and lies on the intersection of a plane (formed the source and
    // basis vectors) and a cone (whose axis is "basis" and whose
    // angle corresponds to cosineOfConeAngle)
    float perpDist = sqrt(1 - (cosineOfConeAngle * cosineOfConeAngle));
	const float3 c0 = float3_scalar_multiply(basis, cosineOfConeAngle);
	const float3 c1 = float3_scalar_multiply(unitPerp, perpDist);
	return float3_scalar_multiply(float3_add(c0, c1), sourceLength);
}

// ----------------------------------------------------------------------------
// Enforce an upper bound on the angle by which a given arbitrary vector
// diviates from a given reference direction (specified by a unit basis
// vector).  The effect is to clip the "source" vector to be inside a cone
// defined by the basis and an angle.
static __inline__ __host__ __device__ float3 limitMaxDeviationAngle (float3 const& source,
                                    const float cosineOfConeAngle,
                                    float3 const& basis)
{
    return vecLimitDeviationAngleUtility (true, // force source INSIDE cone
                                          source,
                                          cosineOfConeAngle,
                                          basis);
}

// ----------------------------------------------------------------------------
// Returns a position randomly distributed on a disk of unit radius
// on the XZ (Y=0) plane, centered at the origin.  Orientation will be
// random and length will range between 0 and 1
static __inline__ __host__ float3 float3_randomVectorOnUnitRadiusXZDisk (void)
{
    float3 v;

    do
    {
		v.x = OpenSteer::frandom01()*2 - 1;
        v.y = 0;
		v.z = OpenSteer::frandom01()*2 - 1;
    }
    while (float3_length(v) >= 1);

    return v;
}

static __inline__ __host__ __device__ float3 float3_LocalRotateForwardToSide(float3 const& v)
{
	return make_float3(-v.z, v.y, v.x);
}

#endif
