#ifndef OPENSTEER_ABSTRACTLOCALSPACE_H
#define OPENSTEER_ABSTRACTLOCALSPACE_H

#include "VectorUtils.cuh"

namespace OpenSteer {
class AbstractLocalSpace
    {
    public:

        // accessors (get and set) for side, up, forward and position
        virtual float3 side (void) const = 0;
        virtual float3 setSide (float3 s) = 0;
        virtual float3 up (void) const = 0;
        virtual float3 setUp (float3 u) = 0;
        virtual float3 forward (void) const = 0;
        virtual float3 setForward (float3 f) = 0;
        virtual float3 position (void) const = 0;
        virtual float3 setPosition (float3 p) = 0;

        // reset transform to identity
        virtual void resetLocalSpace (void) = 0;
		
		// use right-(or left-)handed coordinate space
        virtual bool rightHanded (void) const {return true;}

        // transform a direction in global space to its equivalent in local space
        virtual float3 localizeDirection (const float3& globalDirection) const = 0;

        // transform a point in global space to its equivalent in local space
        virtual float3 localizePosition (const float3& globalPosition) const = 0;

        // transform a point in local space to its equivalent in global space
        virtual float3 globalizePosition (const float3& localPosition) const = 0;

        // transform a direction in local space to its equivalent in global space
        virtual float3 globalizeDirection (const float3& localDirection) const = 0;

        // set "side" basis vector to normalized cross product of forward and up
        virtual void setUnitSideFromForwardAndUp (void) = 0;

        // regenerate the orthonormal basis vectors given a new forward
        // (which is expected to have unit length)
        virtual void regenerateOrthonormalBasisUF (const float3& newUnitForward) = 0;

        // for when the new forward is NOT of unit length
        virtual void regenerateOrthonormalBasis (const float3& newForward) = 0;

        // for supplying both a new forward and and new up
        virtual void regenerateOrthonormalBasis (const float3& newForward,
                                                 const float3& newUp) = 0;

        // rotate 90 degrees in the direction implied by rightHanded()
        virtual float3 localRotateForwardToSide (const float3& v) const = 0;
        virtual float3 globalRotateForwardToSide (const float3& globalForward) const=0;
    };
}

#endif