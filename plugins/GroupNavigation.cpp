#include <iomanip>
#include <sstream>
#include <cstdlib>
#include "OpenSteer/SimpleVehicle.h"
#include "OpenSteer/OpenSteerDemo.h"
    
using namespace OpenSteer;

// ----------------------------------------------------------------------------
// short names for STL vectors (iterators) of SphericalObstacle pointers
typedef std::vector<SphericalObstacle*> SOG;  // spherical obstacle group
typedef SOG::const_iterator SOI;              // spherical obstacle iterator

// ----------------------------------------------------------------------------
// This PlugIn uses two vehicle types: GroupUnit and GroupSheep.  They have a
// common base class: GroupBase which is a specialization of SimpleVehicle.
class GroupBase : public SimpleVehicle
{
public:
    // constructor
    GroupBase () {reset ();}

    // reset state
    void reset (void);

    // draw this character/vehicle into the scene
    void draw (void);

    void randomizeStartingPositionAndHeading (void);
    enum seekerState {running, tagged, atGoal};

    // for draw method
    float3 bodyColor;

    // xxx store steer sub-state for anotation
    bool avoiding;

    // dynamic obstacle registry
    static void initializeObstacles (void);
    static void addOneObstacle (void);
    static void removeOneObstacle (void);
    float minDistanceToObstacle (const float3 point);
    static int obstacleCount;
    static const int maxObstacleCount;
	static const int maxSheepCount;
    static SOG allObstacles;
};

class GroupUnit : public GroupBase
{
public:

    // constructor
    GroupUnit () {reset ();}

    // reset state
    void reset (void);

    // per frame simulation update
    void update (const float currentTime, const float elapsedTime);

    // is there a clear path to the goal?
    bool clearPathToGoal (void);

    float3 steeringForUnit (void);
    void updateState (const float currentTime);
    void draw (void);
    float3 steerToEvadeAllSheep (void);
    float3 XXXsteerToEvadeAllSheep (void);
    void adjustObstacleAvoidanceLookAhead (const bool clearPath);

    seekerState state;
    bool evading; // xxx store steer sub-state for anotation
    float lastRunningTime; // for auto-reset
};


class GroupSheep : public GroupBase
{
public:

    // constructor
    GroupSheep () {reset ();}

    // reset state
    void reset (void);

    // per frame simulation update
    void update (const float currentTime, const float elapsedTime);
};


// ----------------------------------------------------------------------------
// globals
// (perhaps these should be member variables of a Vehicle or PlugIn class)
const int GroupBase::maxObstacleCount = 100;
const int GroupBase::maxSheepCount = 10;

// Min and Max of the start grid
const float3 gStartGridMin = make_float3(-100.0f, 0.0f, -50.0f);
const float3 gStartGridMax = make_float3(-80.0f, 0.0f, 50.0f);

const float3 gStartSheepGridMin = make_float3(-60.0f, 0.0f, -25.0f);
const float3 gStartSheepGridMax = make_float3(60.0f, 0.0f, 25.0f);

const float3 gHomeBaseCenter = make_float3(100.0f, 0.0f, 0.0f);
const float gHomeBaseRadius = 10.0f;

const float gBrakingRate = 0.75;

const float gAvoidancePredictTimeMin  = 0.9f;
const float gAvoidancePredictTimeMax  = 2;
float gAvoidancePredictTime = gAvoidancePredictTimeMin;

// count the number of times the simulation has reset (e.g. for overnight runs)
int resetCount = 0;


// ----------------------------------------------------------------------------
// state for OpenSteerDemo PlugIn
//
const int groupUnitCount = 1000;
GroupUnit * groupUnits[groupUnitCount];

const int groupSheepCount = 10;
GroupSheep * groupSheep[groupSheepCount];


// ----------------------------------------------------------------------------
// reset state
void GroupBase::reset (void)
{
    SimpleVehicle::reset ();  // reset the vehicle 

    setSpeed (3);             // speed along Forward direction.
    setMaxForce (3.0);        // steering force is clipped to this magnitude
    setMaxSpeed (3.0);        // velocity is clipped to this magnitude

    avoiding = false;         // not actively avoiding

    randomizeStartingPositionAndHeading ();  // new starting position

    clearTrailHistory ();     // prevent long streaks due to teleportation
}


void GroupUnit::reset (void)
{
    GroupBase::reset ();
    bodyColor = make_float3(0.4f, 0.4f, 0.6f); // blueish
    state = running;
    evading = false;
}

void GroupSheep::reset (void)
{
    GroupBase::reset ();
    bodyColor = make_float3(0.6f, 0.4f, 0.4f); // redish
}

// ----------------------------------------------------------------------------
// draw this character/vehicle into the scene
void GroupBase::draw (void)
{
    drawBasic2dCircularVehicle (*this, bodyColor);
    drawTrail ();
}


// ----------------------------------------------------------------------------
void GroupBase::randomizeStartingPositionAndHeading (void)
{
	// randomize 2D heading
    randomizeHeadingOnXZPlane ();
}

void GroupSheep::randomizeStartingPositionAndHeading(void)
{
	float xPos = frandom2(

	// are we are too close to an obstacle?
    if (minDistanceToObstacle (position()) < radius()*5)
    {
        // if so, retry the randomization (this recursive call may not return
        // if there is too little free space)
        randomizeStartingPositionAndHeading ();
    }

	return GroupBase::randomizeStartingPositionAndHeading();
}

void GroupUnit::randomizeStartingPositionAndHeading(void)
{
    // randomize position in the starting grid
	float xPos = frandom2(gStartGridMin.x, gStartGridMax.x);
	float zPos = frandom2(gStartGridMin.z, gStartGridMax.z);

	setPosition(make_float3(xPos, 0.0f, zPos));

	return GroupBase::randomizeStartingPositionAndHeading();
}


// ----------------------------------------------------------------------------
void GroupSheep::update (const float currentTime, const float elapsedTime)
{
	//TODO: sheep should avoid obstacles
	// Move randomly
	float3 steer = make_float3(0.0f, 0.0f, 0.0f);

	steer = steerForWander(*this, elapsedTime);
	applySteeringForce(steer, elapsedTime);



 //   // determine upper bound for pursuit prediction time
	//const float seekerToGoalDist = float3_distance(gHomeBaseCenter, gSeeker->position());
 //   const float adjustedDistance = seekerToGoalDist - radius()-gHomeBaseRadius;
 //   const float seekerToGoalTime = ((adjustedDistance < 0 ) ?
 //                                   0 :
 //                                   (adjustedDistance/gSeeker->speed()));
 //   const float maxPredictionTime = seekerToGoalTime * 0.9f;

 //   // determine steering (pursuit, obstacle avoidance, or braking)
 //   float3 steer = make_float3(0, 0, 0);
 //   if (gSeeker->state == running)
 //   {
 //       const float3 avoidance =
 //           steerToAvoidObstacles (*this, gAvoidancePredictTimeMin,
 //                                  (ObstacleGroup&) allObstacles);

 //       // saved for annotation
 //       avoiding = float3_equals(avoidance, float3_zero());

 //       if (avoiding)
 //           steer = steerForPursuit (*this, *gSeeker, maxPredictionTime);
 //       else
 //           steer = avoidance;
 //   }
 //   else
 //   {
 //       applyBrakingForce (gBrakingRate, elapsedTime);
 //   }
 //   applySteeringForce (steer, elapsedTime);

 //   // annotation
 //   annotationVelocityAcceleration ();
 //   recordTrailVertex (currentTime, position());


 //   // detect and record interceptions ("tags") of seeker
 //   const float seekerToMeDist = float3_distance (position(), 
 //                                                gSeeker->position());
 //   const float sumOfRadii = radius() + gSeeker->radius();
 //   if (seekerToMeDist < sumOfRadii)
 //   {
 //       if (gSeeker->state == running) gSeeker->state = tagged;

 //       // annotation:
 //       if (gSeeker->state == tagged)
 //       {
 //           const float3 color = make_float3(0.8f, 0.5f, 0.5f);
 //           annotationXZDisk (sumOfRadii,
	//					float3_scalar_divide(float3_add(position(), gSeeker->position()), 2),
 //                       color,
 //                       20);
 //       }
 //   }
}


// ----------------------------------------------------------------------------
// are there any enemies along the corridor between us and the goal?
bool GroupUnit::clearPathToGoal (void)
{
    const float sideThreshold = radius() * 8.0f;
    const float behindThreshold = radius() * 2.0f;

    const float3 goalOffset = float3_subtract(gHomeBaseCenter, position());
    const float goalDistance = float3_length(goalOffset);
    const float3 goalDirection = float3_scalar_divide(goalOffset, goalDistance);

    const bool goalIsAside = isAside (*this, gHomeBaseCenter, 0.5);

    // for annotation: loop over all and save result, instead of early return 
    bool xxxReturn = true;

    // loop over enemies
    for (int i = 0; i < GroupSheepCount; i++)
    {
        // short name for this enemy
        const GroupSheep& e = *ctfEnemies[i];
        const float eDistance = float3_distance (position(), e.position());
        const float timeEstimate = 0.3f * eDistance / e.speed(); //xxx
        const float3 eFuture = e.predictFuturePosition (timeEstimate);
        const float3 eOffset = float3_subtract(eFuture, position());
        const float alongCorridor = float3_dot(goalDirection, eOffset);
        const bool inCorridor = ((alongCorridor > -behindThreshold) && (alongCorridor < goalDistance));
        const float eForwardDistance = float3_dot(forward(), eOffset);

        // xxx temp move this up before the conditionals
        annotationXZCircle (e.radius(), eFuture, clearPathColor, 20); //xxx

        // consider as potential blocker if within the corridor
        if (inCorridor)
        {
            const float3 perp = float3_subtract(eOffset, float3_scalar_multiply(goalDirection, alongCorridor));
            const float acrossCorridor = float3_length(perp);
            if (acrossCorridor < sideThreshold)
            {
                // not a blocker if behind us and we are perp to corridor
                const float eFront = eForwardDistance + e.radius ();

                //annotationLine (position, forward*eFront, gGreen); // xxx
                //annotationLine (e.position, forward*eFront, gGreen); // xxx

                // xxx
                // std::ostringstream message;
                // message << "eFront = " << std::setprecision(2)
                //         << std::setiosflags(std::ios::fixed) << eFront << std::ends;
                // draw2dTextAt3dLocation (*message.str(), eFuture, gWhite);

                const bool eIsBehind = eFront < -behindThreshold;
                const bool eIsWayBehind = eFront < (-2 * behindThreshold);
                const bool safeToTurnTowardsGoal =
                    ((eIsBehind && goalIsAside) || eIsWayBehind);

                if (! safeToTurnTowardsGoal)
                {
                    // this enemy blocks the path to the goal, so return false
                    annotationLine (position(), e.position(), clearPathColor);
                    // return false;
                    xxxReturn = false;
                }
            }
        }
    }

    // no enemies found along path, return true to indicate path is clear
    // clearPathAnnotation (sideThreshold, behindThreshold, goalDirection);
    // return true;
    if (xxxReturn)
        clearPathAnnotation (sideThreshold, behindThreshold, goalDirection);
    return xxxReturn;
}

float3 GroupUnit::steerToEvadeAllSheep (void)
{
    float3 evade = float3_zero();
    const float goalDistance = float3_distance(gHomeBaseCenter, position());

    // sum up weighted evasion
    for (int i = 0; i < GroupSheepCount; i++)
    {
        const GroupSheep& e = *ctfEnemies[i];
        const float3 eOffset = float3_subtract(e.position(), position());
        const float eDistance = float3_length(eOffset);

        const float eForwardDistance = float3_dot(forward(), eOffset);
        const float behindThreshold = radius() * 2;
        const bool behind = eForwardDistance < behindThreshold;
        if ((!behind) || (eDistance < 5))
        {
            if (eDistance < (goalDistance * 1.2)) //xxx
            {
                const float timeEstimate = 0.5f * eDistance / e.speed();//xxx
                //const float timeEstimate = 0.15f * eDistance / e.speed();//xxx
                const float3 future = e.predictFuturePosition (timeEstimate);

                annotationXZCircle (e.radius(), future, evadeColor, 20); // xxx

                const float3 offset = float3_subtract(future, position());
                const float3 lateral = float3_perpendicularComponent(offset, forward());
                const float d = float3_length(lateral);
                const float weight = -1000 / (d * d);
				evade = float3_add(evade, float3_scalar_multiply(float3_scalar_divide(lateral, d), weight));
            }
        }
    }
    return evade;
}


float3 GroupUnit::XXXsteerToEvadeAllSheep (void)
{
    // sum up weighted evasion
    float3 evade = float3_zero();
    for (int i = 0; i < GroupSheepCount; i++)
    {
        const GroupSheep& e = *ctfEnemies[i];
        const float3 eOffset = float3_subtract(e.position(), position());
        const float eDistance = float3_length(eOffset);

        // xxx maybe this should take into account e's heading? xxx // TODO: just that :)
        const float timeEstimate = 0.5f * eDistance / e.speed(); //xxx
        const float3 eFuture = e.predictFuturePosition (timeEstimate);

        // annotation
        annotationXZCircle (e.radius(), eFuture, evadeColor, 20);

        // steering to flee from eFuture (enemy's future position)
        const float3 flee = xxxsteerForFlee (*this, eFuture);

        const float eForwardDistance = float3_dot(forward(), eOffset);
        const float behindThreshold = radius() * -2;

        const float distanceWeight = 4 / eDistance;
        const float forwardWeight = ((eForwardDistance > behindThreshold) ? 1.0f : 0.5f);

		const float3 adjustedFlee = float3_scalar_multiply(flee, distanceWeight * forwardWeight);

		evade = float3_add(evade, adjustedFlee);
    }
    return evade;
}


// ----------------------------------------------------------------------------


float3 GroupUnit::steeringForUnit (void)
{
    // determine if obstacle avodiance is needed
    const bool clearPath = clearPathToGoal ();
    adjustObstacleAvoidanceLookAhead (clearPath);
    const float3 obstacleAvoidance = steerToAvoidObstacles (*this, gAvoidancePredictTime, (ObstacleGroup&) allObstacles);

    // saved for annotation
    avoiding = !float3_equals(obstacleAvoidance, float3_zero());

    if (avoiding)
    {
        // use pure obstacle avoidance if needed
        return obstacleAvoidance;
    }
    else
    {
        // otherwise seek home base and perhaps evade defenders
        const float3 seek = xxxsteerForSeek (*this, gHomeBaseCenter);
        if (clearPath)
        {
            // we have a clear path (defender-free corridor), use pure seek

            // xxx experiment 9-16-02
            float3 s = limitMaxDeviationAngle (seek, 0.707f, forward());

            annotationLine (position(), float3_add(position(), float3_scalar_multiply(s, 0.2f)), seekColor);
            return s;
        }
        else
        {
            if (1) // xxx testing new evade code xxx
            {
                // combine seek and (forward facing portion of) evasion
                const float3 evade = steerToEvadeAllSheep ();
                const float3 steer = float3_add(seek, limitMaxDeviationAngle (evade, 0.5f, forward()));

                // annotation: show evasion steering force
                annotationLine (position(), float3_add(position(), float3_scalar_multiply(steer, 0.2f)), evadeColor);
                return steer;
            }
            else
            {
                const float3 evade = XXXsteerToEvadeAllSheep ();
                const float3 steer = limitMaxDeviationAngle (float3_add(seek, evade),
                                                           0.707f, forward());

                annotationLine (position(),float3_add(position(),seek), gRed);
                annotationLine (position(),float3_add(position(),evade), gGreen);

                // annotation: show evasion steering force
                annotationLine (position(),float3_add(position(),float3_scalar_multiply(steer,0.2f)),evadeColor);
                return steer;
            }
        }
    }
}


// ----------------------------------------------------------------------------
// adjust obstacle avoidance look ahead time: make it large when we are far
// from the goal and heading directly towards it, make it small otherwise.


void GroupUnit::adjustObstacleAvoidanceLookAhead (const bool clearPath)
{
    if (clearPath)
    {
        evading = false;
        const float goalDistance = float3_distance(gHomeBaseCenter,position());
        const bool headingTowardGoal = isAhead (*this, gHomeBaseCenter, 0.98f);
        const bool isNear = (goalDistance/speed()) < gAvoidancePredictTimeMax;
        const bool useMax = headingTowardGoal && !isNear;
        gAvoidancePredictTime = (useMax ? gAvoidancePredictTimeMax : gAvoidancePredictTimeMin);
    }
    else
    {
        evading = true;
        gAvoidancePredictTime = gAvoidancePredictTimeMin;
    }
}


// ----------------------------------------------------------------------------
void GroupUnit::updateState (const float currentTime)
{
    // if we reach the goal before being tagged, switch to atGoal state
    if (state == running)
    {
        const float baseDistance = float3_distance(position(),gHomeBaseCenter);
        if (baseDistance < (radius() + gHomeBaseRadius))
			state = atGoal;
    }

    // update lastRunningTime (holds off reset time)
    if (state == running)
    {
        lastRunningTime = currentTime;
    }
	else
    {
        const float resetDelay = 4;
        const float resetTime = lastRunningTime + resetDelay;
        if (currentTime > resetTime) 
        {
            // xxx a royal hack (should do this internal to CTF):
            OpenSteerDemo::queueDelayedResetPlugInXXX ();
        }
    }
}


// ----------------------------------------------------------------------------
void GroupUnit::draw (void)
{
    // first call the draw method in the base class
    GroupBase::draw();

    // select string describing current seeker state
    char* seekerStateString = "";
    switch (state)
    {
    case running:
        if (avoiding)
            seekerStateString = "avoid obstacle";
        else if (evading)
            seekerStateString = "seek and evade";
        else
            seekerStateString = "seek goal";
        break;
    case tagged: seekerStateString = "tagged"; break;
    case atGoal: seekerStateString = "reached goal"; break;
    }

    // annote seeker with its state as text
    const float3 textOrigin = float3_add(position(), make_float3(0, 0.25, 0));
    std::ostringstream annote;
    annote << seekerStateString << std::endl;
    annote << std::setprecision(2) << std::setiosflags(std::ios::fixed)
           << speed() << std::ends;
    draw2dTextAt3dLocation (annote, textOrigin, gWhite);

    // display status in the upper left corner of the window
    std::ostringstream status;
    status << seekerStateString << std::endl;
    status << obstacleCount << " obstacles [F1/F2]" << std::endl;
	status << "position: " << position().x << ", " << position().z << std::endl;
    status << resetCount << " restarts" << std::ends;
    const float h = drawGetWindowHeight ();
    const float3 screenLocation = make_float3(10, h-50, 0);
    draw2dTextAt2dLocation (status, screenLocation, gGray80);
}


// ----------------------------------------------------------------------------
// update method for goal seeker
void GroupUnit::update (const float currentTime, const float elapsedTime)
{
    // do behavioral state transitions, as needed
    updateState (currentTime);

    // determine and apply steering/braking forces
    float3 steer = make_float3(0, 0, 0);
    if (state == running)
    {
        steer = steeringForUnit ();
    }
    else
    {
        applyBrakingForce (gBrakingRate, elapsedTime);
    }
    applySteeringForce (steer, elapsedTime);

    // annotation
    annotationVelocityAcceleration ();
    recordTrailVertex (currentTime, position());
}


// ----------------------------------------------------------------------------
// dynamic obstacle registry
//
// xxx need to combine guts of addOneObstacle and minDistanceToObstacle,
// xxx perhaps by having the former call the latter, or change the latter to
// xxx be "nearestObstacle": give it a position, it finds the nearest obstacle
// xxx (but remember: obstacles a not necessarilty spheres!)


int GroupBase::obstacleCount = -1; // this value means "uninitialized"
SOG GroupBase::allObstacles;


#define testOneObstacleOverlap(radius, center)               \
{                                                            \
    float d = float3_distance (c, center);                   \
    float clearance = d - (r + (radius));                    \
    if (minClearance > clearance) minClearance = clearance;  \
}


void GroupBase::initializeObstacles (void)
{
    // start with 40% of possible obstacles
    if (obstacleCount == -1)
    {
        obstacleCount = 0;
        for (int i = 0; i < (maxObstacleCount * 0.4); i++) addOneObstacle ();
    }
}


void GroupBase::addOneObstacle (void)
{
    if (obstacleCount < maxObstacleCount)
    {
        // pick a random center and radius,
        // loop until no overlap with other obstacles and the home base
        float r;
        float3 c;
        float minClearance;
        const float requiredClearance = gSeeker->radius() * 4; // 2 x diameter
        do
        {
            r = frandom2 (1.5, 4);
            c = float3_scalar_multiply(float3_randomVectorOnUnitRadiusXZDisk(), gMaxStartRadius * 1.1f);
            minClearance = FLT_MAX;

            for (SOI so = allObstacles.begin(); so != allObstacles.end(); so++)
            {
                testOneObstacleOverlap ((**so).radius, (**so).center);
            }

            testOneObstacleOverlap (gHomeBaseRadius - requiredClearance,
                                    gHomeBaseCenter);
        }
        while (minClearance < requiredClearance);

        // add new non-overlapping obstacle to registry
        allObstacles.push_back (new SphericalObstacle (r, c));
        obstacleCount++;
    }
}


float GroupBase::minDistanceToObstacle (const float3 point)
{
    float r = 0;
    float3 c = point;
    float minClearance = FLT_MAX;
    for (SOI so = allObstacles.begin(); so != allObstacles.end(); so++)
    {
        testOneObstacleOverlap ((**so).radius, (**so).center);
    }
    return minClearance;
}


void GroupBase::removeOneObstacle (void)
{
    if (obstacleCount > 0)
    {
        obstacleCount--;
        allObstacles.pop_back();
    }
}


// ----------------------------------------------------------------------------
// PlugIn for OpenSteerDemo


class CtfPlugIn : public PlugIn
{
public:

    const char* name (void) {return "Capture the Flag";}

    float selectionOrderSortKey (void) {return 0.01f;}

    virtual ~CtfPlugIn() {} // be more "nice" to avoid a compiler warning

    void open (void)
    {
        // create the seeker ("hero"/"attacker")
        GroupUnit = new GroupUnit;
        all.push_back (GroupUnit);

        // create the specified number of enemies, 
        // storing pointers to them in an array.
        for (int i = 0; i<GroupSheepCount; i++)
        {
            ctfEnemies[i] = new GroupSheep;
            all.push_back (ctfEnemies[i]);
        }

        // initialize camera
        OpenSteerDemo::init2dCamera (*GroupUnit);
        OpenSteerDemo::camera.mode = Camera::cmFixedDistanceOffset;
        OpenSteerDemo::camera.fixedTarget = make_float3(15, 0, 0);
        OpenSteerDemo::camera.fixedPosition = make_float3(80, 60, 0);

        GroupBase::initializeObstacles ();
    }

    void update (const float currentTime, const float elapsedTime)
    {
        // update the seeker
        GroupUnit->update (currentTime, elapsedTime);
      
        // update each enemy
        for (int i = 0; i < GroupSheepCount; i++)
        {
            ctfEnemies[i]->update (currentTime, elapsedTime);
        }
    }

    void redraw (const float currentTime, const float elapsedTime)
    {
        // selected vehicle (user can mouse click to select another)
        AbstractVehicle& selected = *OpenSteerDemo::selectedVehicle;

        // vehicle nearest mouse (to be highlighted)
        AbstractVehicle& nearMouse = *OpenSteerDemo::vehicleNearestToMouse ();

        // update camera
        OpenSteerDemo::updateCamera (currentTime, elapsedTime, selected);

        // draw "ground plane" centered between base and selected vehicle
        const float3 goalOffset = float3_subtract(gHomeBaseCenter, OpenSteerDemo::camera.position());
        const float3 goalDirection = float3_normalize(goalOffset);
        const float3 cameraForward = OpenSteerDemo::camera.xxxls().forward();
        const float goalDot = float3_dot(cameraForward, goalDirection);
        const float blend = remapIntervalClip (goalDot, 1, 0, 0.5, 0);
        const float3 gridCenter = interpolate (blend,
                                             selected.position(),
                                             gHomeBaseCenter);
        OpenSteerDemo::gridUtility (gridCenter);

        // draw the seeker, obstacles and home base
        GroupUnit->draw();
        drawObstacles ();

		drawStart ();
		drawEnd ();

        // draw each enemy
        for (int i = 0; i < groupUnitCount; i++) ctfEnemies[i]->draw ();

        // highlight vehicle nearest mouse
        OpenSteerDemo::highlightVehicleUtility (nearMouse);
    }

    void close (void)
    {
        // delete seeker
        delete (GroupUnit);
        GroupUnit = NULL;

        // delete each enemy
        for (int i = 0; i < GroupSheepCount; i++)
        {
            delete (ctfEnemies[i]);
            ctfEnemies[i] = NULL;
        }

        // clear the group of all vehicles
        all.clear();
    }

    void reset (void)
    {
        // count resets
        resetCount++;

        // reset the seeker ("hero"/"attacker") and enemies
        GroupUnit->reset ();
        for (int i = 0; i<GroupSheepCount; i++) ctfEnemies[i]->reset ();

        // reset camera position
        OpenSteerDemo::position2dCamera (*GroupUnit);

        // make camera jump immediately to new position
        OpenSteerDemo::camera.doNotSmoothNextMove ();
    }

    void handleFunctionKeys (int keyNumber)
    {
        switch (keyNumber)
        {
        case 1: GroupBase::addOneObstacle ();    break;
        case 2: GroupBase::removeOneObstacle (); break;
        }
    }

    void printMiniHelpForFunctionKeys (void)
    {
        std::ostringstream message;
        message << "Function keys handled by ";
        message << '"' << name() << '"' << ':' << std::ends;
        OpenSteerDemo::printMessage (message);
        OpenSteerDemo::printMessage ("  F1     add one obstacle.");
        OpenSteerDemo::printMessage ("  F2     remove one obstacle.");
        OpenSteerDemo::printMessage ("");
    }

    const AVGroup& allVehicles (void) {return (const AVGroup&) all;}

    void drawHomeBase (void)
    {
        const float3 up = make_float3(0, 0.01f, 0);
        const float3 atColor = make_float3(0.3f, 0.3f, 0.5f);
        const float3 noColor = gGray50;
        const bool reached = GroupUnit->state == GroupUnit::atGoal;
        const float3 baseColor = (reached ? atColor : noColor);
        drawXZDisk (gHomeBaseRadius,    gHomeBaseCenter, baseColor, 40);
        drawXZDisk (gHomeBaseRadius/15, float3_add(gHomeBaseCenter, up), gBlack, 20);
    }

    void drawObstacles (void)
    {
        const float3 color = make_float3(0.8f, 0.6f, 0.4f);
        const SOG& allSO = GroupBase::allObstacles;
        for (SOI so = allSO.begin(); so != allSO.end(); so++)
        {
            drawXZCircle ((**so).radius, (**so).center, color, 40);
        }
    }

    // a group (STL vector) of all vehicles in the PlugIn
    std::vector<GroupBase*> all;
};


CtfPlugIn gCtfPlugIn;


// ----------------------------------------------------------------------------


