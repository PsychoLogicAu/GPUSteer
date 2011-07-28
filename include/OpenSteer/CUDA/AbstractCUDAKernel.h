#ifndef OPENSTEER_ABSTRACTCUDAKERNEL_H
#define OPENSTEER_ABSTRACTCUDAKERNEL_H

#include <cuda_runtime.h>

#include "..\AbstractKernel.h"
#include "CUDAGlobals.h"

namespace OpenSteer
{
	const int THREADSPERBLOCK = 128;

	class AbstractCUDAKernel : public AbstractKernel
	{
	protected:
		int m_threadsPerBlock;

		// Device pointers.
		VehicleData*	m_pdVehicleData;
		VehicleConst*	m_pdVehicleConst;

	public:
		AbstractCUDAKernel(VehicleGroup *pVehicleGroup)
		:	AbstractKernel(pVehicleGroup),
			m_pdVehicleData(NULL),
			m_threadsPerBlock(THREADSPERBLOCK)
		{}

		virtual void init(void) = 0;
		virtual void run(void) = 0;
		virtual void close(void) = 0;

		virtual dim3 gridDim(void)
		{
			return dim3((getNumberOfAgents() + m_threadsPerBlock - 1) / m_threadsPerBlock);
		}

		virtual dim3 blockDim(void)
		{
			return dim3(m_threadsPerBlock);
		}
	};
}
#endif