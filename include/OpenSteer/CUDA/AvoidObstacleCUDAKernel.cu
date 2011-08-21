/*

#include "AvoidObstaclesCUDA.h"
#include "CUDAKernelGlobals.cuh"

using namespace OpenSteer;

extern "C"
{
	typedef struct {
		bool intersects;
		float3 position;
		float distance;
	} Intersection;

	__inline__ __device__ bool findNextIntersectionWithSphere(const float3 *center, const float radius, Intersection *intersection)
	{
		float b, c, d, p, q, s;

		// compute line-sphere intersection parameters
		b = -2 * center->z;
		c = center->x * center->x + center->y * center->y + center->z * center->z - radius * radius;
		d = (b * b) - (4 * c);

		// when the path does not intersect the sphere
		if (d < 0)
			return false;

		// otherwise, the path intersects the sphere in two points with
		// parametric coordinates of "p" and "q".
		// (If "d" is zero the two points are coincident, the path is tangent)
		s = sqrt(d);
		p = (-b + s) / 2;
		q = (-b - s) / 2;

		// both intersections are behind us, so no potential collisions
		if ((p < 0) && (q < 0))
			return false;

		// at least one intersection is in front of us
		intersection->intersects = true;
		intersection->distance =
			((p > 0) && (q > 0)) ?
			// both intersections are in front of us, find nearest one
			((p < q) ? p : q) :
			// otherwise only one intersections is in front, select it
			((p > 0) ? p : q);
		return true;
	}

	__inline__ __device__ void LocalizeDirection(const float3 *globalDirection, float3 *localDirection, const vehicle_data *vehicleData)
	{
		// Dot offset with local basis vectors to obtain local coordiantes.
		*localDirection =  make_float3(	float3_dot(*globalDirection, vehicleData->side),
										float3_dot(*globalDirection, vehicleData->up),
										float3_dot(*globalDirection, vehicleData->forward)	);
	}

	__inline__ __device__ void LocalizePosition(const float3 *globalPosition, float3 *localPosition, const vehicle_data *vehicleData)
	{
		// Global offset from local origin.
		float3 globalOffset = float3_subtract(*globalPosition, vehicleData->position);

		LocalizeDirection(&globalOffset, localPosition, vehicleData);
	}

	__global__ void AvoidObstacleCUDAKernel(vehicle_data *vehicleData, vehicle_const *vehicleConst, int numAgents, spherical_obstacle_data *obstacleData, const float minTimeToCollision)
	{
		int offset = (blockIdx.x * blockDim.x) + threadIdx.x;

		// Check bounds.
		if(offset >= numAgents)
			return;

		// If we already have a steering vector set, do nothing.
		if(!float3_equals(STEERING(offset), float3_zero()))
			return;

		const float minDistanceToCollision = minTimeToCollision * SPEED(offset);

		Intersection intersect;
		intersect.intersects = false;

		// Transform the obstacle into local space
		float3 localPosition;
		LocalizePosition(&obstacleData->center, &localPosition, &VDATA(offset));

		// Obstacle is behind the vehicle.
		if(localPosition.z < 0.0f) // TODO: check handedness of this.
		{
			return;
		}

		const float radius = obstacleData->radius;

		// Check whether the obstacle intersects the cylinder
		findNextIntersectionWithSphere(&localPosition, RADIUS(offset) + radius, &intersect);

		// when an intersection was found
		if ((intersect.intersects == true) && (intersect.distance < minDistanceToCollision))
		{
			// compute avoidance steering force: take offset from obstacle to me,
			// take the component of that which is lateral (perpendicular to my
			// forward direction), set length to maxForce, add a bit of forward
			// component (in capture the flag, we never want to slow down)
			const float3 obstacleOffset = float3_subtract(POSITION(offset), intersect.position);
			float3 desiredVelocity = float3_perpendicularComponent(obstacleOffset, FORWARD(offset));
			float3_normalize(desiredVelocity);
			desiredVelocity = float3_scalar_multiply(desiredVelocity, MAXFORCE(offset));
			desiredVelocity = float3_add(desiredVelocity, float3_scalar_multiply(FORWARD(offset), MAXFORCE(offset) * 0.75f));

			STEERING(offset) = desiredVelocity;
		}
	}
}

*/