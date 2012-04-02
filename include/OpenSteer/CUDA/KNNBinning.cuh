#ifndef OPENSTEER_KNNBINNING_CUH
#define OPENSTEER_KNNBINNING_CUH

#include "CUDAGlobals.cuh"

using namespace OpenSteer;

#if defined KNNBINNINGV1
	#include "KNNBinningV1.cuh"

	typedef KNNBinningV1UpdateDBCUDA KNNBinningUpdateDBCUDA;
	typedef KNNBinningV1CUDA KNNBinningCUDA;

#elif defined KNNBINNINGV2
	#include "KNNBinningV2.cuh"

	typedef KNNBinningV2UpdateDBCUDA KNNBinningUpdateDBCUDA;
	typedef KNNBinningV2CUDA KNNBinningCUDA;

#else
	#include "KNNBinningV3.cuh"

	typedef KNNBinningV3UpdateDBCUDA KNNBinningUpdateDBCUDA;
	typedef KNNBinningV3CUDA KNNBinningCUDA;
#endif

#endif
