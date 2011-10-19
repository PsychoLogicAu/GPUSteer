#ifndef OPENSTEER_KNNBINNDATA_CUH
#define OPENSTEER_KNNBINNDATA_CUH

#include "CUDAGlobals.cuh"

using namespace OpenSteer;

#if defined KNNBINNINGV1
	#include "KNNBinDataV1.cuh"

	typedef KNNBinDataV1 KNNBinData;

#else
	#include "KNNBinDataV2.cuh"

	typedef KNNBinDataV2 KNNBinData;
#endif

#endif
