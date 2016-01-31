#include "OpenSteer/SteerLibrary.h"

void
OpenSteer::SteerLibrary::
reset(void)
{
	Annotation::reset();

    // initial state of wander behavior
    WanderSide = 0;
    WanderUp = 0;

    // default to non-gaudyPursuitAnnotation
    gaudyPursuitAnnotation = false;
}

float3
OpenSteer::SteerLibrary::
steerForWander (const AbstractVehicle& v, float dt)
{
    // random walk WanderSide and WanderUp between -1 and +1
    const float speed = 12 * dt; // maybe this (12) should be an argument?
    WanderSide = scalarRandomWalk (WanderSide, speed, -1, +1);
    WanderUp   = scalarRandomWalk (WanderUp,   speed, -1, +1);

    // return a pure lateral steering vector: (+/-Side) + (+/-Up)
    return float3_add(float3_scalar_multiply(v.side(), WanderSide), float3_scalar_multiply(v.up(), WanderUp));
}


// ----------------------------------------------------------------------------
// Seek behavior



float3
OpenSteer::SteerLibrary::
steerForSeek (const AbstractVehicle& v, const float3& target)
{
    const float3 desiredVelocity = float3_subtract(target, make_float3(v.position()));
    return float3_subtract(desiredVelocity, v.velocity());
}


// ----------------------------------------------------------------------------
// Flee behavior



float3
OpenSteer::SteerLibrary::
steerForFlee (const AbstractVehicle& v, const float3& target)
{
    const float3 desiredVelocity = float3_subtract(make_float3(v.position()), target);
    return float3_subtract(desiredVelocity, v.velocity());
}


// ----------------------------------------------------------------------------
// xxx proposed, experimental new seek/flee [cwr 9-16-02]



float3
OpenSteer::SteerLibrary::
xxxsteerForFlee (const AbstractVehicle& v, const float3& target)
{
    const float3 offset = float3_subtract(make_float3(v.position()), target);
    const float3 desiredVelocity = float3_truncateLength(offset, v.maxSpeed());
    return float3_subtract(desiredVelocity, v.velocity());
}



float3
OpenSteer::SteerLibrary::
xxxsteerForSeek (const AbstractVehicle& v, const float3& target)
{
    const float3 offset = float3_subtract(target, make_float3(v.position()));
	const float3 desiredVelocity = float3_truncateLength(offset, v.maxSpeed());
    return float3_subtract(desiredVelocity, v.velocity());
}


// ----------------------------------------------------------------------------
// Path Following behaviors



float3
OpenSteer::SteerLibrary::
steerToStayOnPath (const AbstractVehicle& v, 
				   const float predictionTime, 
				   Pathway& path)
{
    // predict our future position
    const float3 futurePosition = v.predictFuturePosition (predictionTime);

    // find the point on the path nearest the predicted future position
    float3 tangent;
    float outside;
    const float3 onPath = path.mapPointToPath (futurePosition,
                                             tangent,     // output argument
                                             outside);    // output argument

    if (outside < 0)
    {
        // our predicted future position was in the path,
        // return zero steering.
        return float3_zero();
    }
    else
    {
        // our predicted future position was outside the path, need to
        // steer towards it.  Use onPath projection of futurePosition
        // as seek target
        annotatePathFollowing (futurePosition, onPath, onPath, outside);
        return steerForSeek (v, onPath);
    }
}



float3
OpenSteer::SteerLibrary::
steerToFollowPath (const AbstractVehicle& v, 
				   const int direction,
                   const float predictionTime,
                   Pathway& path)
{
    // our goal will be offset from our path distance by this amount
    const float pathDistanceOffset = direction * predictionTime * v.speed();

    // predict our future position
    const float3 futurePosition = v.predictFuturePosition (predictionTime);

    // measure distance along path of our current and predicted positions
    const float nowPathDistance =
        path.mapPointToPathDistance (make_float3(v.position ()));
    const float futurePathDistance =
        path.mapPointToPathDistance (futurePosition);

    // are we facing in the correction direction?
    const bool rightway = ((pathDistanceOffset > 0) ?
                           (nowPathDistance < futurePathDistance) :
                           (nowPathDistance > futurePathDistance));

    // find the point on the path nearest the predicted future position
    // XXX need to improve calling sequence, maybe change to return a
    // XXX special path-defined object which includes two float3s and a 
    // XXX bool (onPath,tangent (ignored), withinPath)
    float3 tangent;
    float outside;
    const float3 onPath = path.mapPointToPath (futurePosition,
                                             // output arguments:
                                             tangent,
                                             outside);

    // no steering is required if (a) our future position is inside
    // the path tube and (b) we are facing in the correct direction
    if ((outside < 0) && rightway)
    {
        // all is well, return zero steering
        return float3_zero();
    }
    else
    {
        // otherwise we need to steer towards a target point obtained
        // by adding pathDistanceOffset to our current path position

        float targetPathDistance = nowPathDistance + pathDistanceOffset;
        float3 target = path.mapPathDistanceToPoint (targetPathDistance);

        annotatePathFollowing (futurePosition, onPath, target, outside);

        // return steering to seek target on path
        return steerForSeek (v, target);
    }
}


// ----------------------------------------------------------------------------
// Obstacle Avoidance behavior
//
// Returns a steering force to avoid a given obstacle.  The purely lateral
// steering force will turn our vehicle towards a silhouette edge of the
// obstacle.  Avoidance is required when (1) the obstacle intersects the
// vehicle's current path, (2) it is in front of the vehicle, and (3) is
// within minTimeToCollision seconds of travel at the vehicle's current
// velocity.  Returns a zero vector value (Vec3::zero) when no avoidance is
// required.
//
// XXX The current (4-23-03) scheme is to dump all the work on the various
// XXX Obstacle classes, making them provide a "steer vehicle to avoid me"
// XXX method.  This may well change.
//
// XXX 9-12-03: this routine is probably obsolete: its name is too close to
// XXX the new steerToAvoidObstacles and the arguments are reversed
// XXX (perhaps there should be another version of steerToAvoidObstacles
// XXX whose second arg is "const Obstacle& obstacle" just in case we want
// XXX to avoid a non-grouped obstacle)


float3
OpenSteer::SteerLibrary::
steerToAvoidObstacle (const AbstractVehicle& v, 
					  const float minTimeToCollision,
                      const Obstacle& obstacle)
{
    const float3 avoidance = obstacle.steerToAvoid (v, minTimeToCollision);

    // XXX more annotation modularity problems (assumes spherical obstacle)
    if (!float3_equals(avoidance, float3_zero()))
        annotateAvoidObstacle (minTimeToCollision * v.speed());

    return avoidance;
}


// this version avoids all of the obstacles in an ObstacleGroup

float3
OpenSteer::SteerLibrary::
steerToAvoidObstacles (const AbstractVehicle& v, 
					   const float minTimeToCollision,
                       ObstacleGroup& obstacles)
{
	return float3_zero();

	/*
    float3 avoidance = float3_zero();
    PathIntersection nearest, next;
    const float minDistanceToCollision = minTimeToCollision * v.speed();

	// Get the collection of near obstacles.
	SphericalObstacleDataVec nearObstacles;
	obstacles.FindNearObstacles(v.position(), 8.0f, nearObstacles);

    next.intersect = false;
    nearest.intersect = false;

    // test all obstacles for intersection with my forward axis,
    // select the one whose point of intersection is nearest
 	for(SphericalObstacleDataIt o = nearObstacles.begin(); o != nearObstacles.end(); o++)
	{
		// xxx this should be a generic call on Obstacle, rather than
		// xxx this code which presumes the obstacle is spherical
		findNextIntersectionWithSphere (v, (**o), next);

		if ((nearest.intersect == false) || ((next.intersect != false) && (next.distance < nearest.distance)))
			nearest = next;
	}

    // when a nearest intersection was found
    if ((nearest.intersect != false) &&
        (nearest.distance < minDistanceToCollision))
    {
        // show the corridor that was checked for collisions
        annotateAvoidObstacle (minDistanceToCollision);

        // compute avoidance steering force: take offset from obstacle to me,
        // take the component of that which is lateral (perpendicular to my
        // forward direction), set length to maxForce, add a bit of forward
        // component (in capture the flag, we never want to slow down)
        const float3 offset = float3_subtract(v.position(), nearest.obstacle->center);

		//float3 steering = float3_perpendicularComponent(obstacleOffset, FORWARD(offset));
		//STEERING(offset) = float3_subtract(steering, FORWARD(offset));

        avoidance = float3_perpendicularComponent(offset, v.forward());
        avoidance = float3_normalize(avoidance);
		avoidance = float3_scalar_multiply(avoidance, v.maxForce());
		avoidance = float3_add(avoidance,  float3_scalar_multiply(v.forward(), v.maxForce () * 0.75f));
    }

    return avoidance;
	*/
}


// ----------------------------------------------------------------------------
// Unaligned collision avoidance behavior: avoid colliding with other nearby
// vehicles moving in unconstrained directions.  Determine which (if any)
// other other vehicle we would collide with first, then steers to avoid the
// site of that potential collision.  Returns a steering force vector, which
// is zero length if there is no impending collision.



float3
OpenSteer::SteerLibrary::
steerToAvoidNeighbors (const AbstractVehicle& v, 
					   const float minTimeToCollision,
                       const AVGroup& others)
{
    // first priority is to prevent immediate interpenetration
    const float3 separation = steerToAvoidCloseNeighbors (v, 0, others);
    
	if (!float3_equals(separation, float3_zero()))
		return separation;

    // otherwise, go on to consider potential future collisions
    float steer = 0;
    AbstractVehicle* threat = NULL;

    // Time (in seconds) until the most immediate collision threat found
    // so far.  Initial value is a threshold: don't look more than this
    // many frames into the future.
    float minTime = minTimeToCollision;

    // xxx solely for annotation
    float3 xxxThreatPositionAtNearestApproach;
    float3 xxxOurPositionAtNearestApproach;

    // for each of the other vehicles, determine which (if any)
    // pose the most immediate threat of collision.
    for (AVIterator i = others.begin(); i != others.end(); i++)
    {
        AbstractVehicle& other = **i;
        if (&other != &v)
        {	
            // avoid when future positions are this close (or less)
            const float collisionDangerThreshold = v.radius() * 2;

            // predicted time until nearest approach of "this" and "other"
            const float time = predictNearestApproachTime (v, other);

            // If the time is in the future, sooner than any other
            // threatened collision...
            if ((time >= 0) && (time < minTime))
            {
                // if the two will be close enough to collide,
                // make a note of it
                if (computeNearestApproachPositions (v, other, time)
                    < collisionDangerThreshold)
                {
                    minTime = time;
                    threat = &other;
                    xxxThreatPositionAtNearestApproach
                        = hisPositionAtNearestApproach;
                    xxxOurPositionAtNearestApproach
                        = ourPositionAtNearestApproach;
                }
            }
        }
    }

    // if a potential collision was found, compute steering to avoid
    if (threat != NULL)
    {
        // parallel: +1, perpendicular: 0, anti-parallel: -1
        float parallelness = float3_dot(make_float3(v.forward()), make_float3(threat->forward()));
        float angle = 0.707f;

        if (parallelness < -angle)
        {
            // anti-parallel "head on" paths:
            // steer away from future threat position
            float3 offset = float3_subtract(xxxThreatPositionAtNearestApproach, make_float3(v.position()));
            float sideDot = float3_dot(offset, v.side());
            steer = (sideDot > 0) ? -1.0f : 1.0f;
        }
        else
        {
            if (parallelness > angle)
            {
                // parallel paths: steer away from threat
                float3 offset = float3_subtract(make_float3(threat->position()), make_float3(v.position()));
                float sideDot = float3_dot(offset, v.side());
                steer = (sideDot > 0) ? -1.0f : 1.0f;
            }
            else
            {
                // perpendicular paths: steer behind threat
                // (only the slower of the two does this)
                if (threat->speed() <= v.speed())
                {
                    float sideDot = float3_dot(v.side(), threat->velocity());
                    steer = (sideDot > 0) ? -1.0f : 1.0f;
                }
            }
        }

        annotateAvoidNeighbor (*threat,
                               steer,
                               xxxOurPositionAtNearestApproach,
                               xxxThreatPositionAtNearestApproach);
    }

	return float3_scalar_multiply(v.side(), steer);
}



// Given two vehicles, based on their current positions and velocities,
// determine the time until nearest approach
//
// XXX should this return zero if they are already in contact?


float
OpenSteer::SteerLibrary::
predictNearestApproachTime (const AbstractVehicle& v, 
							const AbstractVehicle& other)
{
    // imagine we are at the origin with no velocity,
    // compute the relative velocity of the other vehicle
    const float3 myVelocity = v.velocity();
    const float3 otherVelocity = other.velocity();
    const float3 relVelocity = float3_subtract(otherVelocity, myVelocity);
    const float relSpeed = float3_length(relVelocity);

    // for parallel paths, the vehicles will always be at the same distance,
    // so return 0 (aka "now") since "there is no time like the present"
    if (relSpeed == 0)
		return 0;

    // Now consider the path of the other vehicle in this relative
    // space, a line defined by the relative position and velocity.
    // The distance from the origin (our vehicle) to that line is
    // the nearest approach.

    // Take the unit tangent along the other vehicle's path
	const float3 relTangent = float3_scalar_divide(relVelocity, relSpeed);

    // find distance from its path to origin (compute offset from
    // other to us, find length of projection onto path)
    const float3 relPosition = float3_subtract(make_float3(v.position()), make_float3(other.position()));
	const float projection = float3_dot(relTangent, relPosition);

    return projection / relSpeed;
}


// Given the time until nearest approach (predictNearestApproachTime)
// determine position of each vehicle at that time, and the distance
// between them



float
OpenSteer::SteerLibrary::
computeNearestApproachPositions (const AbstractVehicle& v, 
								 AbstractVehicle& other, 
								 float time)
{
	const float3 myTravel = float3_scalar_multiply(make_float3(v.forward()), v.speed () * time);
	const float3 otherTravel = float3_scalar_multiply(make_float3(other.forward()), other.speed () * time);

    const float3 myFinal = float3_add(make_float3(v.position()), myTravel);
    const float3 otherFinal = float3_add(make_float3(other.position()), otherTravel);

    // xxx for annotation
    ourPositionAtNearestApproach = myFinal;
    hisPositionAtNearestApproach = otherFinal;

	return float3_distance(myFinal, otherFinal);
}



// ----------------------------------------------------------------------------
// avoidance of "close neighbors" -- used only by steerToAvoidNeighbors
//
// XXX  Does a hard steer away from any other agent who comes withing a
// XXX  critical distance.  Ideally this should be replaced with a call
// XXX  to steerForSeparation.



float3
OpenSteer::SteerLibrary::
steerToAvoidCloseNeighbors (const AbstractVehicle& v, 
							const float minSeparationDistance,
                            const AVGroup& others)
{
    // for each of the other vehicles...
    for (AVIterator i = others.begin(); i != others.end(); i++)    
    {
        AbstractVehicle& other = **i;
        if (&other != &v)
        {
            const float sumOfRadii = v.radius() + other.radius();
            const float minCenterToCenter = minSeparationDistance + sumOfRadii;
            const float3 offset = float3_subtract(make_float3(other.position()), make_float3(v.position()));
            const float currentDistance = float3_length(offset);

            if (currentDistance < minCenterToCenter)
            {
                annotateAvoidCloseNeighbor (other, minSeparationDistance);
				return float3_perpendicularComponent(float3_minus(offset), make_float3(v.forward()));
            }
        }
    }

    // otherwise return zero
    return float3_zero();
}


// ----------------------------------------------------------------------------
// used by boid behaviors: is a given vehicle within this boid's neighborhood?



bool
OpenSteer::SteerLibrary::
inBoidNeighborhood (const AbstractVehicle& v, 
					const AbstractVehicle& other,
                    const float minDistance,
                    const float maxDistance,
                    const float cosMaxAngle)
{
    if (&other == &v)
    {
        return false;
    }
    else
    {
        const float3 offset = float3_subtract(make_float3(other.position()), make_float3(v.position()));
		const float distanceSquared = float3_lengthSquared(offset);

        // definitely in neighborhood if inside minDistance sphere
        if (distanceSquared < (minDistance * minDistance))
        {
            return true;
        }
        else
        {
            // definitely not in neighborhood if outside maxDistance sphere
            if (distanceSquared > (maxDistance * maxDistance))
            {
                return false;
            }
            else
            {
                // otherwise, test angular offset from forward axis
				const float3 unitOffset = float3_scalar_divide(offset, sqrt(distanceSquared));
                const float forwardness = float3_dot(make_float3(v.forward()), unitOffset);
                return forwardness > cosMaxAngle;
            }
        }
    }
}


// ----------------------------------------------------------------------------
// Separation behavior: steer away from neighbors



float3
OpenSteer::SteerLibrary::
steerForSeparation (const AbstractVehicle& v, 
					const float maxDistance,
                    const float cosMaxAngle,
                    const AVGroup& flock)
{
    // steering accumulator and count of neighbors, both initially zero
    float3 steering = float3_zero();
    int neighbors = 0;

    // for each of the other vehicles...
    for (AVIterator other = flock.begin(); other != flock.end(); other++)
    {
        if (inBoidNeighborhood (v, **other, v.radius() * 3, maxDistance, cosMaxAngle))
        {
            // add in steering contribution
            // (opposite of the offset direction, divided once by distance
            // to normalize, divided another time to get 1/d falloff)
            const float3 offset = float3_subtract(make_float3((**other).position()), make_float3(v.position()));
            const float distanceSquared = float3_dot(offset, offset);
			steering = float3_add(steering, float3_scalar_divide(offset, -distanceSquared));

            // count neighbors
            neighbors++;
        }
    }

    // divide by neighbors, then normalize to pure direction
    if (neighbors > 0) 
		steering = float3_normalize(float3_scalar_divide(steering, (float)neighbors));

    return steering;
}


// ----------------------------------------------------------------------------
// Alignment behavior: steer to head in same direction as neighbors



float3
OpenSteer::SteerLibrary::
steerForAlignment (const AbstractVehicle& v, 
				   const float maxDistance,
                   const float cosMaxAngle,
                   const AVGroup& flock)
{
    // steering accumulator and count of neighbors, both initially zero
    float3 steering = float3_zero();
    int neighbors = 0;

    // for each of the other vehicles...
    for (AVIterator other = flock.begin(); other != flock.end(); other++)
    {
        if (inBoidNeighborhood (v, **other, v.radius() * 3, maxDistance, cosMaxAngle))
        {
            // accumulate sum of neighbor's heading
			steering = float3_add(steering, make_float3((**other).forward()));

            // count neighbors
            neighbors++;
        }
    }

    // divide by neighbors, subtract off current heading to get error-
    // correcting direction, then normalize to pure direction
    if (neighbors > 0)
		steering = float3_normalize(float3_subtract(float3_scalar_divide(steering, (float)neighbors), make_float3(v.forward())));

    return steering;
}


// ----------------------------------------------------------------------------
// Cohesion behavior: to to move toward center of neighbors




float3
OpenSteer::SteerLibrary::
steerForCohesion (const AbstractVehicle& v, 
				  const float maxDistance,
                  const float cosMaxAngle,
                  const AVGroup& flock)
{
    // steering accumulator and count of neighbors, both initially zero
    float3 steering = float3_zero();
    int neighbors = 0;

    // for each of the other vehicles...
    for (AVIterator other = flock.begin(); other != flock.end(); other++)
    {
        if (inBoidNeighborhood (v, **other, v.radius() * 3, maxDistance, cosMaxAngle))
        {
            // accumulate sum of neighbor's positions
			steering = float3_add(steering, make_float3((**other).position()));

            // count neighbors
            neighbors++;
        }
    }

    // divide by neighbors, subtract off current position to get error-
    // correcting direction, then normalize to pure direction
    if (neighbors > 0)
		steering = float3_normalize(float3_subtract(float3_scalar_divide(steering, (float)neighbors), make_float3(v.position())));

    return steering;
}


// ----------------------------------------------------------------------------
// pursuit of another vehicle (& version with ceiling on prediction time)



float3
OpenSteer::SteerLibrary::
steerForPursuit (const AbstractVehicle& v, 
				 const AbstractVehicle& quarry)
{
    return steerForPursuit (v, quarry, FLT_MAX);
}



float3
OpenSteer::SteerLibrary::
steerForPursuit (const AbstractVehicle& v, 
				 const AbstractVehicle& quarry,
                 const float maxPredictionTime)
{
    // offset from this to quarry, that distance, unit vector toward quarry
    const float3 offset = float3_subtract(make_float3(quarry.position()), make_float3(v.position()));
	const float distance = float3_length(offset);
    const float3 unitOffset = float3_scalar_divide(offset, distance);

    // how parallel are the paths of "this" and the quarry
    // (1 means parallel, 0 is pependicular, -1 is anti-parallel)
    const float parallelness = float3_dot(make_float3(v.forward()), make_float3(quarry.forward()));

    // how "forward" is the direction to the quarry
    // (1 means dead ahead, 0 is directly to the side, -1 is straight back)
    const float forwardness = float3_dot(make_float3(v.forward()), unitOffset);

    const float directTravelTime = distance / v.speed ();
    const int f = intervalComparison (forwardness,  -0.707f, 0.707f);
    const int p = intervalComparison (parallelness, -0.707f, 0.707f);

    float timeFactor = 0; // to be filled in below
    float3 color;           // to be filled in below (xxx just for debugging)

    // Break the pursuit into nine cases, the cross product of the
    // quarry being [ahead, aside, or behind] us and heading
    // [parallel, perpendicular, or anti-parallel] to us.
    switch (f)
    {
    case +1:
        switch (p)
        {
        case +1:          // ahead, parallel
            timeFactor = 4;
            color = gBlack;
            break;
        case 0:           // ahead, perpendicular
            timeFactor = 1.8f;
            color = gGray50;
            break;
        case -1:          // ahead, anti-parallel
            timeFactor = 0.85f;
            color = gWhite;
            break;
        }
        break;
    case 0:
        switch (p)
        {
        case +1:          // aside, parallel
            timeFactor = 1;
            color = gRed;
            break;
        case 0:           // aside, perpendicular
            timeFactor = 0.8f;
            color = gYellow;
            break;
        case -1:          // aside, anti-parallel
            timeFactor = 4;
            color = gGreen;
            break;
        }
        break;
    case -1:
        switch (p)
        {
        case +1:          // behind, parallel
            timeFactor = 0.5f;
            color= gCyan;
            break;
        case 0:           // behind, perpendicular
            timeFactor = 2;
            color= gBlue;
            break;
        case -1:          // behind, anti-parallel
            timeFactor = 2;
            color = gMagenta;
            break;
        }
        break;
    }

    // estimated time until intercept of quarry
    const float et = directTravelTime * timeFactor;

    // xxx experiment, if kept, this limit should be an argument
    const float etl = (et > maxPredictionTime) ? maxPredictionTime : et;

    // estimated position of quarry at intercept
    const float3 target = quarry.predictFuturePosition (etl);

    // annotation
    annotationLine (make_float3(v.position()),
                    target,
                    gaudyPursuitAnnotation ? color : gGray40);

    return steerForSeek (v, target);
}

// ----------------------------------------------------------------------------
// evasion of another vehicle



float3
OpenSteer::SteerLibrary::
steerForEvasion (const AbstractVehicle& v, 
				 const AbstractVehicle& menace,
                 const float maxPredictionTime)
{
    // offset from this to menace, that distance, unit vector toward menace
    const float3 offset = float3_subtract(make_float3(menace.position()), make_float3(v.position()));
    const float distance = float3_length(offset);

    const float roughTime = distance / menace.speed();
    const float predictionTime = ((roughTime > maxPredictionTime) ?
                                  maxPredictionTime :
                                  roughTime);

    const float3 target = menace.predictFuturePosition (predictionTime);

    return steerForFlee (v, target);
}


// ----------------------------------------------------------------------------
// tries to maintain a given speed, returns a maxForce-clipped steering
// force along the forward/backward axis



float3
OpenSteer::SteerLibrary::
steerForTargetSpeed (const AbstractVehicle& v, 
					 const float targetSpeed)
{
    const float mf = v.maxForce ();
    const float speedError = targetSpeed - v.speed ();
    return float3_scalar_multiply(make_float3(v.forward ()), clip (speedError, -mf, +mf));
}


// ----------------------------------------------------------------------------
// xxx experiment cwr 9-6-02



void
OpenSteer::SteerLibrary::
findNextIntersectionWithSphere (const AbstractVehicle& v, 
								SphericalObstacleData& obs,
                                PathIntersection& intersection)
{
    // xxx"SphericalObstacle& obs" should be "const SphericalObstacle&
    // obs" but then it won't let me store a pointer to in inside the
    // PathIntersection

    // This routine is based on the Paul Bourke's derivation in:
    //   Intersection of a Line and a Sphere (or circle)
    //   http://www.swin.edu.au/astronomy/pbourke/geometry/sphereline/

    float b, c, d, p, q, s;
    float3 lc;

    // initialize pathIntersection object
    intersection.intersect = false;
    intersection.obstacle = &obs;

    // find "local center" (lc) of sphere in boid's coordinate space
    lc = v.localizePosition (obs.center);

    // computer line-sphere intersection parameters
    b = -2 * lc.z;
    c = square (lc.x) + square (lc.y) + square (lc.z) - 
        square (obs.radius + v.radius());
    d = (b * b) - (4 * c);

    // when the path does not intersect the sphere
    if (d < 0) return;

    // otherwise, the path intersects the sphere in two points with
    // parametric coordinates of "p" and "q".
    // (If "d" is zero the two points are coincident, the path is tangent)
    s = sqrtXXX (d);
    p = (-b + s) / 2;
    q = (-b - s) / 2;

    // both intersections are behind us, so no potential collisions
    if ((p < 0) && (q < 0)) return; 

    // at least one intersection is in front of us
    intersection.intersect = true;
    intersection.distance =
        ((p > 0) && (q > 0)) ?
        // both intersections are in front of us, find nearest one
        ((p < q) ? p : q) :
        // otherwise only one intersections is in front, select it
        ((p > 0) ? p : q);
    return;
}
