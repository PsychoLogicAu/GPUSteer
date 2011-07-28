// ----------------------------------------------------------------------------
//
//
// OpenSteer -- Steering Behaviors for Autonomous Characters
//
// Copyright (c) 2002-2003, Sony Computer Entertainment America
// Original author: Craig Reynolds <craig_reynolds@playstation.sony.com>
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
// THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS IN THE SOFTWARE.
//
//
// ----------------------------------------------------------------------------
//
//
// SteerLibraryMixin
//
// This mixin (class with templated superclass) adds the "steering library"
// functionality to a given base class.  SteerLibraryMixin assumes its base
// class supports the AbstractVehicle interface.
//
// 10-04-04 bk:  put everything into the OpenSteer namespace
// 02-06-03 cwr: create mixin (from "SteerMass")
// 06-03-02 cwr: removed TS dependencies
// 11-21-01 cwr: created
//
//
// ----------------------------------------------------------------------------


#ifndef OPENSTEER_STEERLIBRARY_H
#define OPENSTEER_STEERLIBRARY_H

#include "AbstractVehicle.h"
#include "Annotation.h"
#include "ObstacleGroup.h"
//#include "LocalSpace.h"

#include "Pathway.h"
#include "Obstacle.h"
#include "Utilities.h"
#include "Draw.h"



namespace OpenSteer {

    // ----------------------------------------------------------------------------

	//class SteerLibrary : public AbstractVehicle, public Annotation//, public LocalSpace
	class SteerLibrary : public Annotation
    {
    public:

        // Constructor: initializes state
        SteerLibrary ()
        {
            // set inital state
            reset ();
        }

        // reset state
        void reset (void);

        // -------------------------------------------------- steering behaviors

        // Wander behavior
        float WanderSide;
        float WanderUp;
        float3 steerForWander (const AbstractVehicle& v, float dt);

        // Seek behavior
        float3 steerForSeek (const AbstractVehicle& v, const float3& target);

        // Flee behavior
        float3 steerForFlee (const AbstractVehicle& v, const float3& target);

        // xxx proposed, experimental new seek/flee [cwr 9-16-02]
        float3 xxxsteerForFlee (const AbstractVehicle& v, const float3& target);
        float3 xxxsteerForSeek (const AbstractVehicle& v, const float3& target);

        // Path Following behaviors
        float3 steerToFollowPath (	const AbstractVehicle& v, 
									const int direction,
									const float predictionTime,
									Pathway& path);
        float3 steerToStayOnPath (	const AbstractVehicle& v, 
									const float predictionTime, 
									Pathway& path);

        // ------------------------------------------------------------------------
        // Obstacle Avoidance behavior
        //
        // Returns a steering force to avoid a given obstacle.  The purely
        // lateral steering force will turn our vehicle towards a silhouette edge
        // of the obstacle.  Avoidance is required when (1) the obstacle
        // intersects the vehicle's current path, (2) it is in front of the
        // vehicle, and (3) is within minTimeToCollision seconds of travel at the
        // vehicle's current velocity.  Returns a zero vector value (Vec3::zero)
        // when no avoidance is required.


        float3 steerToAvoidObstacle (	const AbstractVehicle& v, 
										const float minTimeToCollision,
										const Obstacle& obstacle);


        // avoids all obstacles in an ObstacleGroup

        float3 steerToAvoidObstacles (	const AbstractVehicle& v, 
										const float minTimeToCollision,
										ObstacleGroup& obstacles);


        // ------------------------------------------------------------------------
        // Unaligned collision avoidance behavior: avoid colliding with other
        // nearby vehicles moving in unconstrained directions.  Determine which
        // (if any) other other vehicle we would collide with first, then steers
        // to avoid the site of that potential collision.  Returns a steering
        // force vector, which is zero length if there is no impending collision.
        float3 steerToAvoidNeighbors (	const AbstractVehicle& v, 
										const float minTimeToCollision, 
										const AVGroup& others);


        // Given two vehicles, based on their current positions and velocities,
        // determine the time until nearest approach
        float predictNearestApproachTime (	const AbstractVehicle& v, 
											const AbstractVehicle& other);

        // Given the time until nearest approach (predictNearestApproachTime)
        // determine position of each vehicle at that time, and the distance
        // between them
        float computeNearestApproachPositions (	const AbstractVehicle& v, 
												AbstractVehicle& other,
												float time);


        /// XXX globals only for the sake of graphical annotation
        float3 hisPositionAtNearestApproach;
        float3 ourPositionAtNearestApproach;


        // ------------------------------------------------------------------------
        // avoidance of "close neighbors" -- used only by steerToAvoidNeighbors
        //
        // XXX  Does a hard steer away from any other agent who comes withing a
        // XXX  critical distance.  Ideally this should be replaced with a call
        // XXX  to steerForSeparation.


        float3 steerToAvoidCloseNeighbors (	const AbstractVehicle& v, 
											const float minSeparationDistance,
											const AVGroup& others);


        // ------------------------------------------------------------------------
        // used by boid behaviors


        bool inBoidNeighborhood (	const AbstractVehicle& v, 
									const AbstractVehicle& other,
									const float minDistance,
									const float maxDistance,
									const float cosMaxAngle);


        // ------------------------------------------------------------------------
        // Separation behavior -- determines the direction away from nearby boids


        float3 steerForSeparation (	const AbstractVehicle& v, 
									const float maxDistance,
									const float cosMaxAngle,
									const AVGroup& flock);


        // ------------------------------------------------------------------------
        // Alignment behavior

        float3 steerForAlignment (	const AbstractVehicle& v, 
									const float maxDistance,
									const float cosMaxAngle,
									const AVGroup& flock);


        // ------------------------------------------------------------------------
        // Cohesion behavior


        float3 steerForCohesion (	const AbstractVehicle& v, 
									const float maxDistance,
									const float cosMaxAngle,
									const AVGroup& flock);


        // ------------------------------------------------------------------------
        // pursuit of another vehicle (& version with ceiling on prediction time)


        float3 steerForPursuit (	const AbstractVehicle& v, 
									const AbstractVehicle& quarry);

        float3 steerForPursuit (	const AbstractVehicle& v, 
									const AbstractVehicle& quarry,
									const float maxPredictionTime);

        // for annotation
        bool gaudyPursuitAnnotation;


        // ------------------------------------------------------------------------
        // evasion of another vehicle


        float3 steerForEvasion (	const AbstractVehicle& v, 
									const AbstractVehicle& menace,
									const float maxPredictionTime);


        // ------------------------------------------------------------------------
        // tries to maintain a given speed, returns a maxForce-clipped steering
        // force along the forward/backward axis


        float3 steerForTargetSpeed (	const AbstractVehicle& v, 
										const float targetSpeed);

        // ----------------------------------------------------------- utilities
        // XXX these belong somewhere besides the steering library
        // XXX above AbstractVehicle, below SimpleVehicle
        // XXX ("utility vehicle"?)

        // xxx cwr experimental 9-9-02 -- names OK?
        bool isAhead (const AbstractVehicle& v, const float3& target) const {return isAhead (v, target, 0.707f);};
        bool isAside (const AbstractVehicle& v, const float3& target) const {return isAside (v, target, 0.707f);};
        bool isBehind (const AbstractVehicle& v, const float3& target) const {return isBehind (v, target, -0.707f);};

        bool isAhead (const AbstractVehicle& v, const float3& target, float cosThreshold) const
        {
			const float3 targetDirection = float3_normalize(float3_subtract(target, v.position()));
			return float3_dot(v.forward(), targetDirection) > cosThreshold;
        };
        bool isAside (const AbstractVehicle& v, const float3& target, float cosThreshold) const
        {
			const float3 targetDirection = float3_normalize(float3_subtract(target, v.position()));

            const float dp = float3_dot(v.forward(), targetDirection);
            return (dp < cosThreshold) && (dp > -cosThreshold);
        };
        bool isBehind (const AbstractVehicle& v, const float3& target, float cosThreshold) const
        {
            const float3 targetDirection = float3_normalize(float3_subtract(target, v.position()));
            return float3_dot(v.forward(), targetDirection) < cosThreshold;
        };


        // xxx cwr 9-6-02 temporary to support old code
        typedef struct {
            int intersect;
            float distance;
            float3 surfacePoint;
            float3 surfaceNormal;
            SphericalObstacleData* obstacle;
        } PathIntersection;

        // xxx experiment cwr 9-6-02
        void findNextIntersectionWithSphere (	const AbstractVehicle& v, 
												SphericalObstacleData& obs,
												PathIntersection& intersection);


        // ------------------------------------------------ graphical annotation
        // (parameter names commented out to prevent compiler warning from "-W")


        // called when steerToAvoidObstacles decides steering is required
        // (default action is to do nothing, layered classes can overload it)
        virtual void annotateAvoidObstacle (const float /*minDistanceToCollision*/)
        {
        }

        // called when steerToFollowPath decides steering is required
        // (default action is to do nothing, layered classes can overload it)
        virtual void annotatePathFollowing (const float3& /*future*/,
                                            const float3& /*onPath*/,
                                            const float3& /*target*/,
                                            const float /*outside*/)
        {
        }

        // called when steerToAvoidCloseNeighbors decides steering is required
        // (default action is to do nothing, layered classes can overload it)
        virtual void annotateAvoidCloseNeighbor (const AbstractVehicle& /*other*/,
                                                 const float /*additionalDistance*/)
        {
        }

        // called when steerToAvoidNeighbors decides steering is required
        // (default action is to do nothing, layered classes can overload it)
        virtual void annotateAvoidNeighbor (const AbstractVehicle& /*threat*/,
                                            const float /*steer*/,
                                            const float3& /*ourFuture*/,
                                            const float3& /*threatFuture*/)
        {
        }
    };

    
} // namespace OpenSteer

// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
#endif // OPENSTEER_STEERLIBRARY_H
