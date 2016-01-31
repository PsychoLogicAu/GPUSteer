#ifndef OPENSTEER_CUDAUTILS_CUH
#define OPENSTEER_CUDAUTILS_CUH

#include "CUDAGlobals.cuh"

namespace OpenSteer
{

// Swap function.
template <typename T>
__inline__ __device__ __host__ void swap( T & a, T & b )
{
	T temp = a;
	a = b;
	b = temp;
}


}
#endif