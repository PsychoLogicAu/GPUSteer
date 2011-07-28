#include "OpenSteer/Annotation.h"

// ----------------------------------------------------------------------------
// Constructor and destructor
OpenSteer::Annotation::Annotation (void)
{
    trailVertices = NULL;
    trailFlags = NULL;

	reset();

    // xxx I wonder if it makes more sense to NOT do this here, see if the
    // xxx vehicle class calls it to set custom parameters, and if not, set
    // xxx these default parameters on first call to a "trail" function.  The
    // xxx issue is whether it is a problem to allocate default-sized arrays
    // xxx then to free them and allocate new ones
    setTrailParameters (5, 100);  // 5 seconds with 100 points along the trail
}

OpenSteer::Annotation::~Annotation (void)
{
	reset();
}

void OpenSteer::Annotation::reset(void)
{
	if(trailVertices)
		delete[] trailVertices;
	trailVertices = NULL;

	if(trailFlags)
		delete[] trailFlags;
	trailFlags = NULL;
}


// ----------------------------------------------------------------------------
// set trail parameters: the amount of time it represents and the number of
// samples along its length.  re-allocates internal buffers.



void 
OpenSteer::Annotation::setTrailParameters (const float duration, const int vertexCount)
{
    // record new parameters
    trailDuration = duration;
    trailVertexCount = vertexCount;

    // reset other internal trail state
    trailIndex = 0;
    trailLastSampleTime = 0;
    trailSampleInterval = trailDuration / trailVertexCount;
    trailDottedPhase = 1;

    // prepare trailVertices array: free old one if needed, allocate new one
	if(trailVertices)
		delete[] trailVertices;
    trailVertices = new float3[trailVertexCount];

    // prepare trailFlags array: free old one if needed, allocate new one
    if(trailFlags)
		delete[] trailFlags;
    trailFlags = new char[trailVertexCount];

    // initializing all flags to zero means "do not draw this segment"
    for (int i = 0; i < trailVertexCount; i++) trailFlags[i] = 0;
}


// ----------------------------------------------------------------------------
// forget trail history: used to prevent long streaks due to teleportation
//
// XXX perhaps this coudl be made automatic: triggered when the change in
// XXX position is well out of the range of the vehicles top velocity



void 
OpenSteer::Annotation::clearTrailHistory (void)
{
    // brute force implementation, reset everything
    setTrailParameters (trailDuration, trailVertexCount);
}


// ----------------------------------------------------------------------------
// record a position for the current time, called once per update



void 
OpenSteer::Annotation::recordTrailVertex (const float currentTime, const float3 position)
{
    const float timeSinceLastTrailSample = currentTime - trailLastSampleTime;
    if (timeSinceLastTrailSample > trailSampleInterval)
    {
        trailIndex = (trailIndex + 1) % trailVertexCount;
        trailVertices [trailIndex] = position;
        trailDottedPhase = (trailDottedPhase + 1) % 2;
        const int tick = (floorXXX (currentTime) >
                          floorXXX (trailLastSampleTime));
        trailFlags [trailIndex] = trailDottedPhase | (tick ? 2 : 0);
        trailLastSampleTime = currentTime;
    }
    curPosition = position;
}


// ----------------------------------------------------------------------------
// draw the trail as a dotted line, fading away with age



void 
OpenSteer::Annotation::drawTrail (const float3& trailColor, const float3& tickColor)
{
    if (OpenSteerDemo::annotationIsOn())
    {
        int index = trailIndex;
        for (int j = 0; j < trailVertexCount; j++)
        {
            // index of the next vertex (mod around ring buffer)
            const int next = (index + 1) % trailVertexCount;

            // "tick mark": every second, draw a segment in a different color
            const int tick = ((trailFlags [index] & 2) ||
                              (trailFlags [next] & 2));
            const float3 color = tick ? tickColor : trailColor;

            // draw every other segment
            if (trailFlags [index] & 1)
            {
                if (j == 0)
                {
                    // draw segment from current position to first trail point
                    drawLineAlpha (curPosition,
                                   trailVertices [index],
                                   color,
                                   1);
                }
                else
                {
                    // draw trail segments with opacity decreasing with age
                    const float minO = 0.05f; // minimum opacity
                    const float fraction = (float) j / trailVertexCount;
                    const float opacity = (fraction * (1 - minO)) + minO;
                    drawLineAlpha (trailVertices [index],
                                   trailVertices [next],
                                   color,
                                   opacity);
                }
            }
            index = next;
        }
    }
}


// ----------------------------------------------------------------------------
// request (deferred) drawing of a line for graphical annotation
//
// This is called during OpenSteerDemo's simulation phase to annotate behavioral
// or steering state.  When annotation is enabled, a description of the line
// segment is queued to be drawn during OpenSteerDemo's redraw phase.



void 
OpenSteer::Annotation::annotationLine (const float3& startPoint, const float3& endPoint, const float3& color)
{
    if (OpenSteerDemo::annotationIsOn())
    {
        if (OpenSteerDemo::phaseIsDraw())
        {
            drawLine (startPoint, endPoint, color);
        }
        else
        {
            deferredDrawLine (startPoint, endPoint, color);
        }
    }
}


// ----------------------------------------------------------------------------
// request (deferred) drawing of a circle (or disk) for graphical annotation
//
// This is called during OpenSteerDemo's simulation phase to annotate behavioral
// or steering state.  When annotation is enabled, a description of the
// "circle or disk" is queued to be drawn during OpenSteerDemo's redraw phase.



void 
OpenSteer::Annotation::annotationCircleOrDisk (const float radius,
                                                           const float3& axis,
                                                           const float3& center,
                                                           const float3& color,
                                                           const int segments,
                                                           const bool filled,
                                                           const bool in3d)
{
    if (OpenSteerDemo::annotationIsOn())
    {
        if (OpenSteerDemo::phaseIsDraw())
        {
            drawCircleOrDisk (radius, axis, center, color,
                              segments, filled, in3d);
        }
        else
        {
            deferredDrawCircleOrDisk (radius, axis, center, color,
                                      segments, filled, in3d);
        }
    }
}
