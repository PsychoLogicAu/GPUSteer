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
// SimpleVehicle
//
// A steerable point mass with a velocity-aligned local coordinate system.
// SimpleVehicle is useful for developing prototype vehicles in OpenSteerDemo,
// it is the base class for vehicles in the PlugIns supplied with OpenSteer.
// Note that SimpleVehicle is intended only as sample code.  Your application
// can use the OpenSteer library without using SimpleVehicle, as long as you
// implement the AbstractVehicle protocol.
//
// SimpleVehicle makes use of the "mixin" concept from OOP.  To quote
// "Mixin-Based Programming in C++" a clear and helpful paper by Yannis
// Smaragdakis and Don Batory (http://citeseer.nj.nec.com/503808.html):
//
//     ...The idea is simple: we would like to specify an extension without
//     predetermining what exactly it can extend. This is equivalent to
//     specifying a subclass while leaving its superclass as a parameter to be
//     determined later. The benefit is that a single class can be used to
//     express an incremental extension, valid for a variety of classes...
// 
// In OpenSteer, vehicles are defined by an interface: an abstract base class
// called AbstractVehicle.  Implementations of that interface, and related
// functionality (like steering behaviors and vehicle physics) are provided as
// template-based mixin classes.  The intent of this design is to allow you to
// reuse OpenSteer code with your application's own base class.
//
// 10-04-04 bk:  put everything into the OpenSteer namespace
// 01-29-03 cwr: created
//
//
// ----------------------------------------------------------------------------


#ifndef OPENSTEER_SIMPLEVEHICLE_H
#define OPENSTEER_SIMPLEVEHICLE_H


#include "AbstractVehicle.h"
#include "SteerLibrary.h"
#include "VehicleGroupData.cu"

namespace OpenSteer {


    // ----------------------------------------------------------------------------

    class SimpleVehicle : public AbstractVehicle, public SteerLibrary
    {
	protected:
		VehicleData _data;
		VehicleConst _const;

        float		_curvature;
        float3		_lastForward;
        float3		_lastPosition;
        float3		_smoothedPosition;
        float		_smoothedCurvature;
		float3		_smoothedAcceleration;

        // measure path curvature (1/turning-radius), maintain smoothed version
        void measurePathCurvature (const float elapsedTime);
    public:
        // constructor
        SimpleVehicle ();

        // destructor
        ~SimpleVehicle ();

		VehicleData& getVehicleData(void) { return _data; }
		VehicleConst& getVehicleConst(void) { return _const; }

        // reset vehicle state
        void reset (void)
        {
            // reset LocalSpace state
            resetLocalSpace ();

			_data.steering = make_float3(0.0f, 0.0f, 0.0f);

            setMass (1);          // mass (defaults to 1 so acceleration=force)
            setSpeed (0);         // speed along Forward direction.

            setRadius (0.5f);     // size of bounding sphere

            setMaxForce (0.1f);   // steering force is clipped to this magnitude
            setMaxSpeed (1.0f);   // velocity is clipped to this magnitude

            // reset bookkeeping to do running averages of these quanities
            resetSmoothedPosition ();
            resetSmoothedCurvature ();
            resetSmoothedAcceleration ();
        }

		// From LocalSpace.h
        float3 side     (void) const {return _data.side;};
        float3 up       (void) const {return _data.up;};
        float3 forward  (void) const {return _data.forward;};
        float3 position (void) const {return _data.position;};
        float3 setSide     (float3 s) {return _data.side = s;};
        float3 setUp       (float3 u) {return _data.up = u;};
        float3 setForward  (float3 f) {return _data.forward = f;};
        float3 setPosition (float3 p) {return _data.position = p;};
        float3 setSide     (float x, float y, float z){return _data.side = make_float3(x,y,z);};
        float3 setUp       (float x, float y, float z){return _data.up = make_float3(x,y,z);};
        float3 setForward  (float x, float y, float z){return _data.forward = make_float3(x,y,z);};
        float3 setPosition (float x, float y, float z){return _data.position = make_float3(x,y,z);};
		bool rightHanded (void) const {return true;}
		void resetLocalSpace (void)
        {
            _data.forward = make_float3 (0, 0, 1);
            _data.side = localRotateForwardToSide (_data.forward);
            _data.up = make_float3 (0, 1, 0);
            _data.position = make_float3 (0, 0, 0);
        };
#pragma region From LocalSpace
		// ------------------------------------------------------------------------
        // transform a direction in global space to its equivalent in local space
        float3 localizeDirection (const float3& globalDirection) const
        {
            // dot offset with local basis vectors to obtain local coordiantes
            return make_float3 (float3_dot(globalDirection, _data.side),
                         float3_dot(globalDirection, _data.up),
                         float3_dot(globalDirection, _data.forward));
        };


        // ------------------------------------------------------------------------
        // transform a point in global space to its equivalent in local space
        float3 localizePosition (const float3& globalPosition) const
        {
            // global offset from local origin
            float3 globalOffset = float3_subtract(globalPosition, _data.position);

            // dot offset with local basis vectors to obtain local coordiantes
            return localizeDirection (globalOffset);
        };


        // ------------------------------------------------------------------------
        // transform a point in local space to its equivalent in global space
        float3 globalizePosition (const float3& localPosition) const
        {
            return float3_add(_data.position, globalizeDirection(localPosition));
        };


        // ------------------------------------------------------------------------
        // transform a direction in local space to its equivalent in global space
        float3 globalizeDirection (const float3& localDirection) const
        {
            return float3_add(
								float3_add(
											float3_scalar_multiply(_data.side, localDirection.x),
											float3_scalar_multiply(_data.up, localDirection.y)
										  ),
								float3_scalar_multiply(_data.forward, localDirection.z)
							 );
        };


        // ------------------------------------------------------------------------
        // set "side" basis vector to normalized cross product of forward and up
        void setUnitSideFromForwardAndUp (void)
        {
            // derive new unit side basis vector from forward and up
            if (rightHanded())
				_data.side = float3_cross(_data.forward, _data.up);
            else
				_data.side = float3_normalize(float3_cross(_data.up, _data.forward));
        }


        // ------------------------------------------------------------------------
        // regenerate the orthonormal basis vectors given a new forward
        // (which is expected to have unit length)
        void regenerateOrthonormalBasisUF (const float3& newUnitForward)
        {
            _data.forward = newUnitForward;

            // derive new side basis vector from NEW forward and OLD up
            setUnitSideFromForwardAndUp ();

            // derive new Up basis vector from new Side and new Forward
            // (should have unit length since Side and Forward are
            // perpendicular and unit length)
            if (rightHanded())
				_data.up = float3_cross(_data.side, _data.forward);
            else
				_data.up = float3_cross(_data.forward, _data.side);
        }


        // for when the new forward is NOT know to have unit length
        void regenerateOrthonormalBasis (const float3& newForward)
        {
			regenerateOrthonormalBasisUF (float3_normalize(newForward));
        }

        // for supplying both a new forward and and new up
        void regenerateOrthonormalBasis (const float3& newForward,
                                         const float3& newUp)
        {
            _data.up = newUp;
            regenerateOrthonormalBasis (float3_normalize(newForward));
        }


        // ------------------------------------------------------------------------
        // rotate, in the canonical direction, a vector pointing in the
        // "forward" (+Z) direction to the "side" (+/-X) direction
        float3 localRotateForwardToSide (const float3& v) const
        {
            return make_float3(rightHanded() ? -v.z : +v.z,
                         v.y,
                         v.x);
        }

        // not currently used, just added for completeness
        float3 globalRotateForwardToSide (const float3& globalForward) const
        {
            const float3 localForward = localizeDirection (globalForward);
            const float3 localSide = localRotateForwardToSide (localForward);
            return globalizeDirection (localSide);
        }
#pragma endregion


        // get/set mass
        float mass (void) const {return _const.mass;}
        float setMass (float m) {return _const.mass = m;}

        // get velocity of vehicle
		float3 velocity (void) const { return _data.velocity(); } //{return float3_scalar_multiply(forward(), _data.speed);}

        // get/set speed of vehicle  (may be faster than taking mag of velocity)
        float speed (void) const {return _data.speed;}
        float setSpeed (float s) {return _data.speed = s;}

        // size of bounding sphere, for obstacle avoidance, etc.
        float radius (void) const {return _const.radius;}
        float setRadius (float m) {return _const.radius = m;}

        // get/set maxForce
        float maxForce (void) const {return _const.maxForce;}
        float setMaxForce (float mf) {return _const.maxForce = mf;}

        // get/set maxSpeed
        float maxSpeed (void) const {return _const.maxSpeed;}
        float setMaxSpeed (float ms) {return _const.maxSpeed = ms;}


        // apply a given steering force to our momentum,
        // adjusting our orientation to maintain velocity-alignment.
        void applySteeringForce (const float3& force, const float deltaTime);

        // the default version: keep FORWARD parallel to velocity, change
        // UP as little as possible.
        virtual void regenerateLocalSpace (const float3& newVelocity,
                                           const float elapsedTime);

        // alternate version: keep FORWARD parallel to velocity, adjust UP
        // according to a no-basis-in-reality "banking" behavior, something
        // like what birds and airplanes do.  (XXX experimental cwr 6-5-03)
        void regenerateLocalSpaceForBanking (const float3& newVelocity,
                                             const float elapsedTime);

        // adjust the steering force passed to applySteeringForce.
        // allows a specific vehicle class to redefine this adjustment.
        // default is to disallow backward-facing steering at low speed.
        // xxx experimental 8-20-02
        virtual float3 adjustRawSteeringForce (const float3& force,
                                             const float deltaTime);

        // apply a given braking force (for a given dt) to our momentum.
        // xxx experimental 9-6-02
        void applyBrakingForce (const float rate, const float deltaTime);

        // predict position of this vehicle at some time in the future
        // (assumes velocity remains constant)
        float3 predictFuturePosition (const float predictionTime) const;

        // get instantaneous curvature (since last update)
        float curvature (void) {return _curvature;}

        // get/reset smoothedCurvature, smoothedAcceleration and smoothedPosition
        float smoothedCurvature (void) {return _smoothedCurvature;}
        float resetSmoothedCurvature (float value = 0)
        {
            _lastForward = float3_zero();
            _lastPosition = float3_zero();
            return _smoothedCurvature = _curvature = value;
        }
        float3 smoothedAcceleration (void) {return _smoothedAcceleration;}
        float3 resetSmoothedAcceleration (const float3& value = float3_zero())
        {
            return _smoothedAcceleration = value;
        }
        float3 smoothedPosition (void) {return _smoothedPosition;}
        float3 resetSmoothedPosition (const float3& value = float3_zero())
        {
            return _smoothedPosition = value;
        }

        // give each vehicle a unique number
        int serialNumber;
        static int serialNumberCounter;

        // draw lines from vehicle's position showing its velocity and acceleration
        void annotationVelocityAcceleration (float maxLengthA, float maxLengthV);
        void annotationVelocityAcceleration (float maxLength)
            {annotationVelocityAcceleration (maxLength, maxLength);}
        void annotationVelocityAcceleration (void)
            {annotationVelocityAcceleration (3, 3);}

        // set a random "2D" heading: set local Up to global Y, then effectively
        // rotate about it by a random angle (pick random forward, derive side).
        void randomizeHeadingOnXZPlane (void)
        {
			setUp (float3_up());
            setForward (float3_RandomUnitVectorOnXZPlane ());
            setSide (localRotateForwardToSide (forward()));
        }
    };

	/*inline void randomizeHeadingOnXZPlane(VehicleData &vehicleData)
    {
		vehicleData.up = float3_up();
		vehicleData.forward = float3_RandomUnitVectorOnXZPlane ();
		vehicleData.side = make_float3(-vehicleData.forward.z, vehicleData.forward.y, vehicleData.forward.x);
    }*/

	inline void randomizeHeadingOnXZPlane( float3 & up, float3 & forward, float3 & side )
    {
		up = float3_up();
		forward = float3_RandomUnitVectorOnXZPlane ();
		side = make_float3( -forward.z, forward.y, forward.x );
    }



} // namespace OpenSteer
    
    
// ----------------------------------------------------------------------------
#endif // OPENSTEER_SIMPLEVEHICLE_H
