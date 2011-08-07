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

static void HandleError( cudaError_t err, const char *file, int line )
{
    if (err != cudaSuccess)
	{
		char buffer[1024];
		sprintf_s(buffer, 1024, "%s in %s at line %d\n", cudaGetErrorString( err ), file, line);
		OutputDebugStringA(buffer);
		MessageBoxA(NULL, buffer, "CUDA Error", MB_OK | MB_ICONERROR);
        exit( EXIT_FAILURE );
    }
}

#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

#endif