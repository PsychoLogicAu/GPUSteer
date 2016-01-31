#ifndef OPENSTEER_TIMER_H
#define	OPENSTEER_TIMER_H

#include <windows.h>

namespace OpenSteer
{
class Timer
{
  LARGE_INTEGER lFreq, lStart;

public:
  Timer()
  {
    QueryPerformanceFrequency(&lFreq);
  }

  inline void Start()
  {
    QueryPerformanceCounter(&lStart);
  }
  
  inline double Stop()
  {
    // Return duration in milliseconds...
    LARGE_INTEGER lEnd;
    QueryPerformanceCounter(&lEnd);
    return (double(lEnd.QuadPart - lStart.QuadPart) / lFreq.QuadPart * 1000);
  }
};	// class Timer
}	// namespace OpenSteer

#endif
