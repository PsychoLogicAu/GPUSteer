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
// camera control for OpenSteerDemo
//
// a camera ("point of view") with various "aiming modes" to track a
// moving vehicle
//
// 10-04-04 bk:  put everything into the OpenSteer namespace
// 06-26-02 cwr: created 
//
//
// ----------------------------------------------------------------------------


#include "OpenSteer/Camera.h"
#include "OpenSteer/OpenSteerDemo.h"


// ----------------------------------------------------------------------------
// constructor


OpenSteer::Camera::Camera (void)
{
    reset ();
}


// ----------------------------------------------------------------------------
// reset all camera state to default values


void 
OpenSteer::Camera::reset (void)
{
    // reset camera's position and orientation
    resetLocalSpace ();

    // "look at" point, center of view
    target = float3_zero();

    // vehicle being tracked
    vehicleToTrack = NULL;

    // aim at predicted position of vehicleToTrack, this far into thefuture
    aimLeadTime = 1;

    // make first update abrupt
    smoothNextMove = false;

    // relative rate at which camera transitions proceed
    smoothMoveSpeed = 1.5f;

    // select camera aiming mode
    mode = cmFixed;

    // "constant distance from vehicle" camera mode parameters
    fixedDistDistance = 1;
    fixedDistVOffset = 0;

    // "look straight down at vehicle" camera mode parameters
    lookdownDistance = 100;

    // "static" camera mode parameters
	//fixedPosition = make_float3(75,75,75);
	fixedPosition = make_float3(0,0,500);
    fixedTarget = float3_zero();
    fixedUp = float3_up();

    // "fixed local offset" camera mode parameters
	fixedLocalOffset = make_float3(5,5,-5);

    // "offset POV" camera mode parameters
    povOffset = make_float3(0, 1, -3);
}


// ----------------------------------------------------------------------------
// called once per frame to update camera state according to currently
// selected mode and per-mode parameters.  Works in position/target/up
// ("look at") space.
//
// parameter names commented out to prevent compiler warning from "-W"


void 
OpenSteer::Camera::update (const float /*currentTime*/,
                        const float elapsedTime,
                        const bool simulationPaused)
{
    // vehicle being tracked (just a reference with a more concise name)
    const AbstractVehicle& v = *vehicleToTrack;
    const bool noVehicle = vehicleToTrack == NULL;
    
    // new position/target/up, set in switch below, defaults to current
    float3 newPosition = make_float3(position());
    float3 newTarget = target;
    float3 newUp = up();


    // prediction time to compensate for lag caused by smoothing moves
    const float antiLagTime = simulationPaused ? 0 : 1 / smoothMoveSpeed;

    // aim at a predicted future position of the target vehicle
    const float predictionTime = aimLeadTime + antiLagTime;

    // set new position/target/up according to camera aim mode
    switch (mode)
    {
    case cmFixed:
        newPosition = fixedPosition;
        newTarget = fixedTarget;
        newUp = fixedUp;
        break;

    case cmFixedDistanceOffset:
        if (noVehicle) break;
        newUp = float3_up(); // xxx maybe this should be v.up ?
        newTarget = v.predictFuturePosition (predictionTime);
        newPosition = constDistHelper (elapsedTime);
        break;

    case cmStraightDown:
        if (noVehicle) break;
        newUp = make_float3(v.forward());
        newTarget = v.predictFuturePosition (predictionTime);
        newPosition = newTarget;
        newPosition.y += lookdownDistance;
        break;

    case cmFixedLocalOffset:
        if (noVehicle) break;
        newUp = v.up();
        newTarget = v.predictFuturePosition (predictionTime);
        newPosition = v.globalizePosition (fixedLocalOffset);
        break;

    case cmOffsetPOV:
        {
            if (noVehicle) break;
            newUp = v.up();
            const float3 futurePosition = v.predictFuturePosition (antiLagTime);
            const float3 globalOffset = v.globalizeDirection (povOffset);
            newPosition = float3_add(futurePosition, globalOffset);
            // XXX hack to improve smoothing between modes (no effect on aim)
            const float L = 10;
            newTarget = float3_add(newPosition , float3_scalar_multiply(make_float3(v.forward()), L));
            break;
        }
    default:
        break;
    }

    // blend from current position/target/up towards new values
    smoothCameraMove (newPosition, newTarget, newUp, elapsedTime);

    // set camera in draw module
    drawCameraLookAt (make_float3(position()), target, up());
}


// ----------------------------------------------------------------------------
// Smoothly move camera: blend (at a rate controlled by smoothMoveSpeed)
// from current camera state toward newly determined camera state.
//
// The flag smoothNextMove can be set (with doNotSmoothNextMove()) to
// make next update (say at simulation initialization time).


void 
OpenSteer::Camera::smoothCameraMove (const float3& newPosition,
                                     const float3& newTarget,
                                     const float3& newUp,
                                     const float elapsedTime)
{
    if (smoothNextMove)
    {
        const float smoothRate = elapsedTime * smoothMoveSpeed;

        float3 tempPosition = make_float3(position());
        float3 tempUp = up();
        blendIntoAccumulator (smoothRate, newPosition, tempPosition);
        blendIntoAccumulator (smoothRate, newTarget,   target);
        blendIntoAccumulator (smoothRate, newUp,       tempUp);
        setPosition (make_float4(tempPosition, 0.f));
        setUp (tempUp);

        // xxx not sure if these are needed, seems like a good idea
        // xxx (also if either up or oldUP are zero, use the other?)
        // xxx (even better: force up to be perp to target-position axis))
        if (float3_equals(up(), float3_zero()))
            setUp (float3_up());
        else
            setUp (float3_normalize(up()));
    }
    else
    {
        smoothNextMove = true;

        setPosition (make_float4(newPosition, 0.f));
        target   = newTarget;
        setUp (newUp);
    }
}


// ----------------------------------------------------------------------------
// computes a new camera position which follows "target" at distant of
// "dragTargetDistance"
//
// parameter names commented out to prevent compiler warning from "-W"


float3
OpenSteer::Camera::constDistHelper (const float /*elapsedTime*/)
{
    // is the "global up"/"vertical" offset constraint enabled?  (it forces
    // the camera's global-up (Y) cordinate to be a above/below the target
    // vehicle by a given offset.)
    const bool constrainUp = (fixedDistVOffset != 0);

    // vector offset from target to current camera position
    const float3 adjustedPosition = make_float3(position().x,
                                 (constrainUp) ? target.y : position().y,
                                 position().z);
    const float3 offset = float3_subtract(adjustedPosition, target);

    // current distance between them
    const float distance = float3_length(offset);

    // move camera only when geometry is well-defined (avoid degenerate case)
    if (distance == 0)
    {
        return make_float3(position());
    }
    else
    {
        // unit vector along original offset
        const float3 unitOffset = float3_scalar_divide(offset, distance);

        // new offset of length XXX
        const float xxxDistance = sqrtXXX (square (fixedDistDistance) -
                                           square (fixedDistVOffset));
        const float3 newOffset = float3_scalar_multiply(unitOffset, xxxDistance);

        // return new camera position: adjust distance to target
		return float3_add(float3_add(target ,newOffset), make_float3(0, fixedDistVOffset, 0));
    }
}


// ----------------------------------------------------------------------------
// select next camera mode, used by OpenSteerDemo


void 
OpenSteer::Camera::selectNextMode (void)
{
    mode = successorMode (mode);
    if (mode >= cmEndMode) mode = successorMode (cmStartMode);
}


// ----------------------------------------------------------------------------
// cycles through the various camera modes


OpenSteer::Camera::cameraMode 
OpenSteer::Camera::successorMode (const cameraMode cm) const
{
    return (cameraMode)(((int)cm) + 1);
}


// ----------------------------------------------------------------------------
// string naming current camera mode, used by OpenSteerDemo


char* 
OpenSteer::Camera::modeName (void)
{
    switch (mode)
    {
    case  cmFixed:               return "static";                break;
    case  cmFixedDistanceOffset: return "fixed distance offset"; break;
    case  cmFixedLocalOffset:    return "fixed local offset";    break;
    case  cmOffsetPOV:           return "offset POV";            break;
    case  cmStraightDown:        return "straight down";         break;
    default:                     return "?";
    }
}


// ----------------------------------------------------------------------------
// adjust the offest vector of the current camera mode based on a
// "mouse adjustment vector" from OpenSteerDemo (xxx experiment 10-17-02)


void 
OpenSteer::Camera::mouseAdjustOffset (const float3& adjustment)
{
    // vehicle being tracked (just a reference with a more concise name)
    const AbstractVehicle& v = *vehicleToTrack;

    switch (mode)
    {
    case cmFixed:
        {
            const float3 offset = float3_subtract(fixedPosition, fixedTarget);
            const float3 adjusted = mouseAdjustPolar (adjustment, offset);
            fixedPosition = float3_add(fixedTarget, adjusted);
            break;
        }
    case cmFixedDistanceOffset:
        {
            // XXX this is the oddball case, adjusting "position" instead
            // XXX of mode parameters, hence no smoothing during adjustment
            // XXX Plus the fixedDistVOffset feature complicates things
            const float3 offset = float3_subtract(make_float3(position()), target);
            const float3 adjusted = mouseAdjustPolar (adjustment, offset);
            // XXX --------------------------------------------------
//             position = target + adjusted;
//             fixedDistDistance = adjusted.length();
//             fixedDistVOffset = position.y - target.y;
            // XXX --------------------------------------------------
//             const float s = smoothMoveSpeed * (1.0f/40f);
//             const Vec3 newPosition = target + adjusted;
//             position = interpolate (s, position, newPosition);
//             fixedDistDistance = interpolate (s, fixedDistDistance, adjusted.length());
//             fixedDistVOffset = interpolate (s, fixedDistVOffset, position.y - target.y);
            // XXX --------------------------------------------------

            setPosition (make_float4(float3_add(target, adjusted), 0.f));
            fixedDistDistance = float3_length(adjusted);

            fixedDistVOffset = position().y - target.y;
            // XXX --------------------------------------------------
            break;
        }
    case cmStraightDown:
        {
            const float3 offset = make_float3(0, 0, lookdownDistance);
            const float3 adjusted = mouseAdjustPolar (adjustment, offset);
            lookdownDistance = adjusted.z;
            break;
        }
    case cmFixedLocalOffset:
        {
            const float3 offset = v.globalizeDirection (fixedLocalOffset);
            const float3 adjusted = mouseAdjustPolar (adjustment, offset);
            fixedLocalOffset = v.localizeDirection (adjusted);
            break;
        }
    case cmOffsetPOV:
        {
            // XXX this might work better as a translation control, it is
            // XXX non-obvious using a polar adjustment when the view
            // XXX center is not at the camera aim target
            const float3 offset = v.globalizeDirection (povOffset);
            const float3 adjusted = mouseAdjustOrtho (adjustment, offset);
            povOffset = v.localizeDirection (adjusted);
            break;
        }
    default:
        break;
    }
}


// ----------------------------------------------------------------------------


float3
OpenSteer::Camera::mouseAdjust2 (const bool polar,
                                 const float3& adjustment,
                                 const float3& offsetToAdjust)
{
    // value to be returned
    float3 result = offsetToAdjust;

    // using the camera's side/up axes (essentially: screen space) move the
    // offset vector sideways according to adjustment.x and vertically
    // according to adjustment.y, constrain the offset vector's length to
    // stay the same, hence the offset's "tip" stays on the surface of a
    // sphere.
    const float oldLength = float3_length(result);
    const float rate = polar ? oldLength : 1;
    //result += xxxls().side() * (adjustment.x * rate);
	result = float3_add(result, float3_scalar_multiply(xxxls().side(), adjustment.x * rate));

    //result += xxxls().up()   * (adjustment.y * rate);
	result = float3_add(result, float3_scalar_multiply(xxxls().up(), adjustment.y * rate));
    if (polar)
    {
        const float newLength = float3_length(result);
		result = float3_scalar_multiply(result, oldLength / newLength);
    }

    // change the length of the offset vector according to adjustment.z
    if (polar)
		result = float3_scalar_multiply(result, 1 + adjustment.z);
    else
		result = float3_add(result, float3_scalar_multiply(make_float3(xxxls().forward()), adjustment.z));

    return result;
}


// ----------------------------------------------------------------------------
