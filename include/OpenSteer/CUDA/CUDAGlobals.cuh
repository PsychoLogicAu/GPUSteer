#ifndef CUDA_GLOBALS_H
#define CUDA_GLOBALS_H

#ifndef _WINDOWS_
	#define WIN32_LEAN_AND_MEAN
	#include <windows.h>
#endif

#include <stdio.h>

#include <cutil_inline.h>

#include <cuda_runtime.h>
#pragma comment(lib, "cudart.lib")

#include "CUDAUtils.cuh"

typedef unsigned int uint;

#endif