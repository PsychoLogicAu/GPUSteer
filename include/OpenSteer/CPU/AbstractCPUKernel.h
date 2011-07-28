#ifndef OPENSTEER_ABSTRACTCPUKERNEL_H
#define OPENSTEER_ABSTRACTCPUKERNEL_H

#include <cuda_runtime.h>
#include "AbstractKernel.h"

// TODO: is there any need for this class or should CPU kernels inherit directly from AbstractKernel?

namespace OpenSteer
{
	class AbstractCUDAKernel : public AbstractKernel
	{
	protected:
	public:
		virtual void init(void) = 0;
		virtual void run(void) = 0;
		virtual void close(void) = 0;
	};
}
#endif