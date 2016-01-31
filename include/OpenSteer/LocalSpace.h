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
// LocalSpace: a local coordinate system for 3d space
//
// Provide functionality such as transforming from local space to global
// space and vice versa.  Also regenerates a valid space from a perturbed
// "forward vector" which is the basis of abnstract vehicle turning.
//
// These are comparable to a 4x4 homogeneous transformation matrix where the
// 3x3 (R) portion is constrained to be a pure rotation (no shear or scale).
// The rows of the 3x3 R matrix are the basis vectors of the space.  They are
// all constrained to be mutually perpendicular and of unit length.  The top
// ("x") row is called "side", the middle ("y") row is called "up" and the
// bottom ("z") row is called forward.  The translation vector is called
// "position".  Finally the "homogeneous column" is always [0 0 0 1].
//
//     [ R R R  0 ]      [ Sx Sy Sz  0 ]
//     [ R R R  0 ]      [ Ux Uy Uz  0 ]
//     [ R R R  0 ]  ->  [ Fx Fy Fz  0 ]
//     [          ]      [             ]
//     [ T T T  1 ]      [ Tx Ty Tz  1 ]
//
// This file defines three classes:
//   AbstractLocalSpace:  pure virtual interface
//   LocalSpaceMixin:     mixin to layer LocalSpace functionality on any base
//   LocalSpace:          a concrete object (can be instantiated)
//
// 10-04-04 bk:  put everything into the OpenSteer namespace
// 06-05-02 cwr: created 
//
//
// ----------------------------------------------------------------------------


#ifndef OPENSTEER_LOCALSPACE_H
#define OPENSTEER_LOCALSPACE_H

#include "VectorUtils.cuh"
#include "AbstractLocalSpace.h"


// ----------------------------------------------------------------------------


namespace OpenSteer {
    // ----------------------------------------------------------------------------
    // LocalSpaceMixin is a mixin layer, a class template with a paramterized base
    // class.  Allows "LocalSpace-ness" to be layered on any class.

    class LocalSpace : public AbstractLocalSpace
    {
        // transformation as three orthonormal unit basis vectors and the
        // origin of the local space.  These correspond to the "rows" of
        // a 3x4 transformation matrix with [0 0 0 1] as the final column

    private:
        float3 _side;     //    side-pointing unit basis vector
        float3 _up;       //  upward-pointing unit basis vector
        float4 _forward;  // forward-pointing unit basis vector
        float4 _position; // origin of local space

    public:

        // accessors (get and set) for side, up, forward and position
        float3 side     (void) const {return _side;};
        float3 up       (void) const {return _up;};
        float4 forward  (void) const {return _forward;};
        float4 position (void) const {return _position;};
        float3 setSide     (float3 s) {return _side = s;};
        float3 setUp       (float3 u) {return _up = u;};
        float4 setForward  (float4 f) {return _forward = f;};
        float4 setPosition (float4 p) {return _position = p;};
        float3 setSide     (float x, float y, float z){return _side = make_float3(x,y,z);};
        float3 setUp       (float x, float y, float z){return _up = make_float3(x,y,z);};
        float4 setForward  (float x, float y, float z){return _forward = make_float4(x,y,z,0);};
        float4 setPosition (float x, float y, float z){return _position = make_float4(x,y,z,0);};


        // ------------------------------------------------------------------------
        // Global compile-time switch to control handedness/chirality: should
        // LocalSpace use a left- or right-handed coordinate system?  This can be
        // overloaded in derived types (e.g. vehicles) to change handedness.

        bool rightHanded (void) const {return true;}


        // ------------------------------------------------------------------------
        // constructors


        LocalSpace (void)
        {
            resetLocalSpace ();
        };

        LocalSpace		 (const float3& Side,
                         const float3& Up,
                         const float4& Forward,
                         const float4& Position)
        {
            _side = Side;
            _up = Up;
            _forward = Forward;
            _position = Position;
        };


        LocalSpace (const float3& Up,
                         const float4& Forward,
                         const float4& Position)
        {
            _up = Up;
            _forward = Forward;
            _position = Position;
            setUnitSideFromForwardAndUp ();
        };


        // ------------------------------------------------------------------------
        // reset transform: set local space to its identity state, equivalent to a
        // 4x4 homogeneous transform like this:
        //
        //     [ X 0 0 0 ]
        //     [ 0 1 0 0 ]
        //     [ 0 0 1 0 ]
        //     [ 0 0 0 1 ]
        //
        // where X is 1 for a left-handed system and -1 for a right-handed system.

        void resetLocalSpace (void)
        {
            _forward = make_float4 (0, 0, 1, 0);
            _side = localRotateForwardToSide (make_float3(_forward));
            _up = make_float3 (0, 1, 0);
            _position = make_float4 (0, 0, 0, 0);
        };


        // ------------------------------------------------------------------------
        // transform a direction in global space to its equivalent in local space


        float3 localizeDirection (const float3& globalDirection) const
        {
            // dot offset with local basis vectors to obtain local coordiantes
            return make_float3 (float3_dot(globalDirection, _side),// globalDirection.dot (_side),
                         float3_dot(globalDirection, _up),// globalDirection.dot (_up),
                         float3_dot(globalDirection, make_float3(_forward)));// globalDirection.dot (_forward));
        };


        // ------------------------------------------------------------------------
        // transform a point in global space to its equivalent in local space


        float3 localizePosition (const float3& globalPosition) const
        {
            // global offset from local origin
            float3 globalOffset = float3_subtract(globalPosition, make_float3(_position));// globalPosition - _position;

            // dot offset with local basis vectors to obtain local coordiantes
            return localizeDirection (globalOffset);
        };


        // ------------------------------------------------------------------------
        // transform a point in local space to its equivalent in global space


        float3 globalizePosition (const float3& localPosition) const
        {
            return float3_add(make_float3(_position), globalizeDirection(localPosition));// _position + globalizeDirection (localPosition);
        };


        // ------------------------------------------------------------------------
        // transform a direction in local space to its equivalent in global space


        float3 globalizeDirection (const float3& localDirection) const
        {
            return float3_add(
								float3_add(
											float3_scalar_multiply(_side, localDirection.x),
											float3_scalar_multiply(_up, localDirection.y)
										  ),
								float3_scalar_multiply(make_float3(_forward), localDirection.z)
							 );
        };


        // ------------------------------------------------------------------------
        // set "side" basis vector to normalized cross product of forward and up


        void setUnitSideFromForwardAndUp (void)
        {
            // derive new unit side basis vector from forward and up
            if (rightHanded())
				_side = float3_cross(make_float3(_forward), _up);
            else
				_side = float3_normalize(float3_cross(_up, make_float3(_forward)));
        }


        // ------------------------------------------------------------------------
        // regenerate the orthonormal basis vectors given a new forward
        // (which is expected to have unit length)


        void regenerateOrthonormalBasisUF (const float3& newUnitForward)
        {
            _forward = make_float4( newUnitForward, 0.f );

            // derive new side basis vector from NEW forward and OLD up
            setUnitSideFromForwardAndUp ();

            // derive new Up basis vector from new Side and new Forward
            // (should have unit length since Side and Forward are
            // perpendicular and unit length)
            if (rightHanded())
				_up = float3_cross(_side, make_float3(_forward));
            else
				_up = float3_cross(make_float3(_forward), _side);
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
            _up = newUp;
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
    };


    //// ----------------------------------------------------------------------------
    //// Concrete LocalSpace class, and a global constant for the identity transform


    //typedef LocalSpaceMixin<AbstractLocalSpace> LocalSpace;

    //const LocalSpace gGlobalSpace;
	const LocalSpace gGlobalSpace;

} // namespace OpenSteer

// ----------------------------------------------------------------------------
#endif // OPENSTEER_LOCALSPACE_H
