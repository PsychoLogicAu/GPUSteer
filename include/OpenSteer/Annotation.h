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
// AnnotationMixin
//
// This mixin (class with templated superclass) adds OpenSteerDemo-based
// graphical annotation functionality to a given base class, which is
// typically something that supports the AbstractVehicle interface.
//
// 10-04-04 bk:  put everything into the OpenSteer namespace
// 04-01-03 cwr: made into a mixin
// 07-01-02 cwr: created (as Annotation.h) 
//
//
// ----------------------------------------------------------------------------


#ifndef OPENSTEER_ANNOTATION_H
#define OPENSTEER_ANNOTATION_H


#include "OpenSteerDemo.h"
// ----------------------------------------------------------------------------


namespace OpenSteer {
	class Annotation
    {
    public:

        // constructor
        Annotation();

        // destructor
        virtual ~Annotation ();

		void reset(void);

        // ------------------------------------------------------------------------
        // trails / streamers
        //
        // these routines support visualization of a vehicle's recent path
        //
        // XXX conceivable trail/streamer should be a separate class,
        // XXX Annotation would "has-a" one (or more))

        // record a position for the current time, called once per update
        void recordTrailVertex (const float currentTime, const float3 position);

        // draw the trail as a dotted line, fading away with age
        void drawTrail (void) {drawTrail (grayColor (0.7f), gWhite);}
        void drawTrail  (const float3& trailColor, const float3& tickColor);

        // set trail parameters: the amount of time it represents and the
        // number of samples along its length.  re-allocates internal buffers.
        void setTrailParameters (const float duration, const int vertexCount);

        // forget trail history: used to prevent long streaks due to teleportation
        void clearTrailHistory (void);

        // ------------------------------------------------------------------------
        // drawing of lines, circles and (filled) disks to annotate steering
        // behaviors.  When called during OpenSteerDemo's simulation update phase,
        // these functions call a "deferred draw" routine which buffer the
        // arguments for use during the redraw phase.
        //
        // note: "circle" means unfilled
        //       "disk" means filled
        //       "XZ" means on a plane parallel to the X and Z axes (perp to Y)
        //       "3d" means the circle is perpendicular to the given "axis"
        //       "segments" is the number of line segments used to draw the circle

        // draw an opaque colored line segment between two locations in space
        void annotationLine (const float3& startPoint,
                             const float3& endPoint,
                             const float3& color);

        // draw a circle on the XZ plane
        void annotationXZCircle (const float radius,
                                 const float3& center,
                                 const float3& color,
                                 const int segments)
        {
            annotationXZCircleOrDisk (radius, center, color, segments, false);
        }


        // draw a disk on the XZ plane
        void annotationXZDisk (const float radius,
                               const float3& center,
                               const float3& color,
                               const int segments)
        {
            annotationXZCircleOrDisk (radius, center, color, segments, true);
        }


        // draw a circle perpendicular to the given axis
        void annotation3dCircle (const float radius,
                                 const float3& center,
                                 const float3& axis,
                                 const float3& color,
                                 const int segments)
        {
            annotation3dCircleOrDisk (radius, center, axis, color, segments, false);
        }


        // draw a disk perpendicular to the given axis
        void annotation3dDisk (const float radius,
                               const float3& center,
                               const float3& axis,
                               const float3& color,
                               const int segments)
        {
            annotation3dCircleOrDisk (radius, center, axis, color, segments, true);
        }

        //

        // ------------------------------------------------------------------------
        // support for annotation circles

        void annotationXZCircleOrDisk (const float radius,
                                       const float3& center,
                                       const float3& color,
                                       const int segments,
                                       const bool filled)
        {
            annotationCircleOrDisk (radius,
                                    float3_zero(),
                                    center,
                                    color,
                                    segments,
                                    filled,
                                    false); // "not in3d" -> on XZ plane
        }


        void annotation3dCircleOrDisk (const float radius,
                                       const float3& center,
                                       const float3& axis,
                                       const float3& color,
                                       const int segments,
                                       const bool filled)
        {
            annotationCircleOrDisk (radius,
                                    axis,
                                    center,
                                    color,
                                    segments,
                                    filled,
                                    true); // "in3d"
        }

        void annotationCircleOrDisk (const float radius,
                                     const float3& axis,
                                     const float3& center,
                                     const float3& color,
                                     const int segments,
                                     const bool filled,
                                     const bool in3d);

        // ------------------------------------------------------------------------
    private:

        // trails
        int trailVertexCount;       // number of vertices in array (ring buffer)
        int trailIndex;             // array index of most recently recorded point
        float trailDuration;        // duration (in seconds) of entire trail
        float trailSampleInterval;  // desired interval between taking samples
        float trailLastSampleTime;  // global time when lat sample was taken
        int trailDottedPhase;       // dotted line: draw segment or not
        float3 curPosition;           // last reported position of vehicle
        float3* trailVertices;        // array (ring) of recent points along trail
        char* trailFlags;           // array (ring) of flag bits for trail points
    };

} // namespace OpenSteer

// ----------------------------------------------------------------------------
#endif // OPENSTEER_ANNOTATION_H
