/*

#include "AvoidObstaclesCUDA.h"

#include "CUDAKernelGlobals.h"

using namespace OpenSteer;

extern "C"
{
	typedef struct {
		bool intersects;
		float distance;
		float3 position;
	} PathIntersection;

	__device__ void findNextIntersectionWithSphere(vehicle_data *vehicleData, vehicle_const *vehicleConst, spherical_obstacle_data *obs, PathIntersection *intersection)
	{
		// This routine is based on the Paul Bourke's derivation in:
		//   Intersection of a Line and a Sphere (or circle)
		//   http://www.swin.edu.au/astronomy/pbourke/geometry/sphereline/

		float b, c, d, p, q, s;
		float3 lc;

		// initialize pathIntersection object
		intersection->intersects = false;
		intersection->position = obs->center;

		// find "local center" (lc) of sphere in boid's coordinate space
		float3 globalOffset = float3_subtract(obs->center, vehicleData->position);
		lc = make_float3(	float3_dot(globalOffset, vehicleData->side),
							float3_dot(globalOffset, vehicleData->up),
							float3_dot(globalOffset, vehicleData->forward)	);

		// computer line-sphere intersection parameters
		b = -2 * lc.z;
		c = lc.x * lc.x + lc.y * lc.y + lc.z * lc.z - 
			(obs->radius + vehicleConst->radius) * (obs->radius + vehicleConst->radius);
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

	__global__ void AvoidObstaclesCUDAKernel(vehicle_data *vehicleData, vehicle_const *vehicleConst, int numAgents, spherical_obstacle_data *obstacleData, near_obstacle_index *nearObstacleIndices, int *obstacleIndices, const float minTimeToCollision)
	{
		int offset = (blockIdx.x * blockDim.x) + threadIdx.x;

		// Check bounds.
		if(offset >= numAgents)
			return;

		// If we already have a steering vector set, do nothing.
		if(!float3_equals(STEERING(offset), float3_zero()))
			return;

		// If there was no obstacle near this agent, do nothing.
		if(nearObstacleIndices[offset].numObstacles == 0)
			return;
		
		// Declare shared memory to store the produced steering vectors.
		//__shared__ float3 desiredVelocity[THREADSPERBLOCK];

		const float minDistanceToCollision = minTimeToCollision * SPEED(offset);
		PathIntersection next, nearest;

		next.intersects = false;
		nearest.intersects = false;

		// TODO: test the effect of __syncthreads() here
		//__syncthreads();

		// test all obstacles for intersection with vehicleData[offset]'s forward axis,
		// select the one whose point of intersection is nearest
		for(int i = nearObstacleIndices[offset].baseIndex; i < nearObstacleIndices[offset].baseIndex + nearObstacleIndices[offset].numObstacles; i++)
		{
			findNextIntersectionWithSphere(&VDATA(offset), &VCONST(offset), &obstacleData[obstacleIndices[i]], &next);

			// Take the dot product of the forward vector with the position of intersection.  Will be positive if ahead of agent.
			//float dotProduct = float3_dot(FORWARD(offset), float3_subtract(next.position, POSITION(offset)));
			float dotProduct = 1.0f;

			if (dotProduct > 0.0f && (nearest.intersects == false) || ((next.intersects != false) && (next.distance < nearest.distance)))
				nearest = next;
		}

		// TODO: test the effect of __syncthreads() here
		//__syncthreads();

		// when a nearest intersection was found
		if ((nearest.intersects != false) && (nearest.distance < minDistanceToCollision))
		{
			// compute avoidance steering force: take offset from obstacle to me,
			// take the component of that which is lateral (perpendicular to my
			// forward direction), set length to maxForce, add a bit of forward
			// component (in capture the flag, we never want to slow down)
			const float3 obstacleOffset = float3_subtract(POSITION(offset), nearest.position);

			float3 steering = float3_perpendicularComponent(obstacleOffset, FORWARD(offset));
			STEERING(offset) = float3_subtract(steering, FORWARD(offset));


			//avoidance = offset.perpendicularComponent (forward());
			//avoidance = avoidance.normalize ();
			//avoidance *= maxForce ();
			//avoidance += forward() * maxForce () * 0.75;
		}
	}
}

*/